"""
Flask Shipping Route Optimizer (single-file)

How to run:
1. Create a virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install flask

2. Run:
   python flask_shipping_optimizer.py

3. Open http://127.0.0.1:5000 in your browser.

What it does (simple):
- Has a small network of ports with lat/lon and adjacency
- Takes user inputs: vessel capacity (tons), port handling costs (per-port override),
  origin, destination, delivery deadline (hours), vessel speed (km/h)
- Enumerates simple paths from origin -> destination (depth-limited), computes
  travel time and cost for each path, filters by deadline, and returns the
  cheapest path meeting the deadline.
- Shows result in a web dashboard and renders a map (Leaflet) with the route.

Notes:
- This is intentionally simple (no ML). Cost model is illustrative: edge cost =
  distance * fuel_price_per_km * (vessel_capacity_factor) + node handling costs
- You can extend: replace path enumeration with more advanced optimization
  (mixed-integer programming, Clarke-Wright, VRP solvers, etc.)

"""
from flask import Flask, request, render_template_string, jsonify
import math
from itertools import permutations, islice

app = Flask(__name__)

# ---------------------- Sample data ----------------------
# Small set of ports (id, name, lat, lon)
PORTS = {
    'IDX': {'name': 'Incheon', 'lat': 37.4563, 'lon': 126.7052},
    'HKG': {'name': 'Hong Kong', 'lat': 22.3193, 'lon': 114.1694},
    'SIN': {'name': 'Singapore', 'lat': 1.3521, 'lon': 103.8198},
    'CNS': {'name': 'CNS', 'lat': 23.1167, 'lon': 113.25},
    'SHA': {'name': 'Shanghai', 'lat': 31.2304, 'lon': 121.4737},
    'PTY': {'name': 'Panama City', 'lat': 8.9833, 'lon': -79.5167},
    'LAX': {'name': 'Los Angeles', 'lat': 33.7405, 'lon': -118.2775},
    'DXB': {'name': 'Dubai', 'lat': 25.2532, 'lon': 55.3657},
}

# Build adjacency by allowing edges between ports that are realistic sea routes.
# For simplicity we'll connect most ports but keep graph small.
ADJ = {
    'IDX': ['HKG', 'SHA'],
    'HKG': ['IDX', 'SIN', 'SHA', 'DXB'],
    'SIN': ['HKG', 'SHA', 'PTY'],
    'SHA': ['IDX', 'HKG', 'SIN', 'DXB'],
    'PTY': ['SIN', 'LAX'],
    'LAX': ['PTY', 'DXB'],
    'DXB': ['HKG', 'SHA', 'LAX'],
    'CNS': ['HKG', 'SHA'],
}

# Parameters
FUEL_PRICE_PER_KM = 0.12  # arbitrary currency per km (per capacity factor)
BASE_HANDLING_COST = 500  # default per port handling cost
DEFAULT_VESSEL_SPEED_KMH = 20  # average speed

# --------------- Utilities ----------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def compute_edge_info(a, b):
    pa = PORTS[a]
    pb = PORTS[b]
    dist = haversine_km(pa['lat'], pa['lon'], pb['lat'], pb['lon'])
    # travel time in hours at default speed
    time_h = dist / DEFAULT_VESSEL_SPEED_KMH
    return {'distance_km': dist, 'time_h': time_h}

# Precompute edges
EDGES = {}
for u, nbrs in ADJ.items():
    for v in nbrs:
        key = tuple(sorted((u, v)))
        if key not in EDGES:
            EDGES[key] = compute_edge_info(u, v)

# --------------- Path enumeration + cost model ----------------

def enumerate_simple_paths(graph, source, dest, max_hops=6):
    # simple DFS (iterative) to enumerate simple paths up to max_hops edges
    stack = [(source, [source])]
    while stack:
        (node, path) = stack.pop()
        if len(path)-1 > max_hops:
            continue
        for nbr in graph.get(node, []):
            if nbr in path:
                continue
            new_path = path + [nbr]
            if nbr == dest:
                yield new_path
            else:
                stack.append((nbr, new_path))


def path_cost_and_time(path, vessel_capacity, port_handling_overrides, speed_kmh, fuel_price_per_km):
    total_cost = 0.0
    total_time = 0.0
    # node handling costs (exclude origin for clarity if desired)
    for node in path:
        handling = port_handling_overrides.get(node, BASE_HANDLING_COST)
        total_cost += handling
    # edges
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        key = tuple(sorted((a, b)))
        info = EDGES.get(key)
        if not info:
            # if no precomputed edge, compute direct
            info = compute_edge_info(a, b)
        dist = info['distance_km']
        # cost scales mildly with vessel capacity (simplified)
        capacity_factor = max(0.5, vessel_capacity / 1000.0)  # normalize
        edge_cost = dist * fuel_price_per_km * capacity_factor
        total_cost += edge_cost
        # time uses chosen speed
        edge_time = dist / speed_kmh
        total_time += edge_time
    return {'cost': round(total_cost, 2), 'time_h': round(total_time, 2)}

# --------------- Flask routes ----------------

INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Shipping Route Optimizer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>body{font-family: Arial, sans-serif; margin:20px} #map{height:400px; margin-top:10px}</style>
  </head>
  <body>
    <h2>Shipping Route Optimizer — Simple</h2>
    <form id="optForm" method="post" action="/optimize">
      <label>Origin: <select name="origin">{% for k,v in ports.items() %}<option value="{{k}}">{{k}} — {{v.name}}</option>{% endfor %}</select></label>
      <label> Destination: <select name="destination">{% for k,v in ports.items() %}<option value="{{k}}">{{k}} — {{v.name}}</option>{% endfor %}</select></label>
      <br><br>
      <label>Vessel capacity (tons): <input name="capacity" type="number" value="1000" min="10" required></label>
      <label>Vessel speed (km/h): <input name="speed" type="number" value="20" min="1"></label>
      <label>Delivery deadline (hours): <input name="deadline" type="number" value="1000" min="1"></label>
      <br><br>
      <label>Fuel price per km: <input name="fuel_price" type="number" value="0.12" step="0.01"></label>
      <br><br>
      <strong>Per-port handling cost overrides (optional, comma-separated CODE:cost):</strong>
      <br>
      <input name="overrides" style="width:80%" placeholder="e.g. HKG:600,SIN:700">
      <br><br>
      <button type="submit">Find cheapest route</button>
    </form>

    {% if result %}
      <h3>Best route (cost {{result.cost}} , time {{result.time_h}} h)</h3>
      <div>Path: {{result.path|join(' → ')}}</div>
      <div id="map"></div>
      <h4>All feasible paths considered</h4>
      <table border="1" cellpadding="6">
        <tr><th>Path</th><th>Cost</th><th>Time (h)</th></tr>
        {% for p in all %}
          <tr><td>{{p.path|join(' → ')}}</td><td>{{p.cost}}</td><td>{{p.time_h}}</td></tr>
        {% endfor %}
      </table>
    {% endif %}

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    {% if result %}
    <script>
      const path = {{result_js|safe}};
      const ports = {{ports_js|safe}};
      const map = L.map('map').setView([0,0], 2);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 18}).addTo(map);
      const latlngs = path.path.map(code => [ports[code].lat, ports[code].lon]);
      const poly = L.polyline(latlngs, {weight:4}).addTo(map);
      // markers
      for (let i=0;i<path.path.length;i++){
        const code = path.path[i];
        L.marker([ports[code].lat, ports[code].lon]).addTo(map).bindPopup(code + ' — ' + ports[code].name);
      }
      map.fitBounds(poly.getBounds(), {padding:[40,40]});
    </script>
    {% endif %}
  </body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML, ports=PORTS, result=None)

@app.route('/optimize', methods=['POST'])
def optimize():
    origin = request.form['origin']
    destination = request.form['destination']
    capacity = float(request.form.get('capacity', 1000))
    speed = float(request.form.get('speed', DEFAULT_VESSEL_SPEED_KMH))
    deadline = float(request.form.get('deadline', 1e9))
    fuel_price = float(request.form.get('fuel_price', FUEL_PRICE_PER_KM))
    overrides_raw = request.form.get('overrides','').strip()
    overrides = {}
    if overrides_raw:
        try:
            for part in overrides_raw.split(','):
                k,v = part.split(':')
                overrides[k.strip().upper()] = float(v)
        except Exception:
            pass

    # enumerate paths
    candidates = []
    for path in enumerate_simple_paths(ADJ, origin, destination, max_hops=6):
        info = path_cost_and_time(path, capacity, overrides, speed, fuel_price)
        # enforce deadline
        if info['time_h'] <= deadline:
            candidates.append({'path': path, 'cost': info['cost'], 'time_h': info['time_h']})

    # sort by cost
    candidates_sorted = sorted(candidates, key=lambda x: x['cost'])
    best = candidates_sorted[0] if candidates_sorted else None

    # Prepare JS-safe objects
    import json
    result_js = json.dumps(best) if best else 'null'
    ports_js = json.dumps(PORTS)

    return render_template_string(INDEX_HTML, ports=PORTS, result=best, all=candidates_sorted, result_js=result_js, ports_js=ports_js)

# Simple API endpoint too
@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    data = request.json or {}
    origin = data.get('origin')
    destination = data.get('destination')
    capacity = float(data.get('capacity', 1000))
    speed = float(data.get('speed', DEFAULT_VESSEL_SPEED_KMH))
    deadline = float(data.get('deadline', 1e9))
    fuel_price = float(data.get('fuel_price', FUEL_PRICE_PER_KM))
    overrides = data.get('overrides', {})

    if origin not in PORTS or destination not in PORTS:
        return jsonify({'error': 'invalid ports'}), 400

    candidates = []
    for path in enumerate_simple_paths(ADJ, origin, destination, max_hops=6):
        info = path_cost_and_time(path, capacity, overrides, speed, fuel_price)
        if info['time_h'] <= deadline:
            candidates.append({'path': path, 'cost': info['cost'], 'time_h': info['time_h']})
    candidates_sorted = sorted(candidates, key=lambda x: x['cost'])
    best = candidates_sorted[0] if candidates_sorted else None
    return jsonify({'best': best, 'all': candidates_sorted})

if __name__ == '__main__':
    app.run(debug=True)
