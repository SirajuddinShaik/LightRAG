import pipmaster as pm

# Install required packages
required_packages = ["networkx", "jinja2"]

for package in required_packages:
    if not pm.is_installed(package):
        pm.install(package)

import networkx as nx
import json
import webbrowser
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict
import random

class CytoscapeGraphMLViewer:
    def __init__(self):
        self.graph = None
        self.node_colors = {
            'organization': '#e74c3c',  # Red
            'person': '#3498db',        # Blue  
            'category': '#2ecc71',      # Green
            'event': '#f39c12',         # Orange
            'default': '#9b59b6'        # Purple
        }
        
    def load_graphml(self, file_path=None):
        """Load GraphML file with file dialog if no path provided"""
        if not file_path:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select GraphML File",
                filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")]
            )
            root.destroy()
            
        if not file_path:
            return False
            
        try:
            self.graph = nx.read_graphml(file_path)
            print(f"Successfully loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
        except Exception as e:
            print(f"Error loading GraphML file: {e}")
            return False
    
    def analyze_graph(self):
        """Analyze the graph structure and provide insights"""
        if not self.graph:
            print("No graph loaded")
            return
            
        print("\n=== GRAPH ANALYSIS ===")
        print(f"Nodes: {self.graph.number_of_nodes()}")
        print(f"Edges: {self.graph.number_of_edges()}")
        # Handle both directed and undirected graphs
        if self.graph.is_directed():
            print(f"Weakly Connected Components: {nx.number_weakly_connected_components(self.graph)}")
            print(f"Strongly Connected Components: {nx.number_strongly_connected_components(self.graph)}")
        else:
            print(f"Connected Components: {nx.number_connected_components(self.graph)}")
        
        # Entity type distribution
        entity_types = defaultdict(int)
        for node_id, data in self.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'unknown')
            entity_types[entity_type] += 1
            
        print("\n=== ENTITY TYPE DISTRIBUTION ===")
        for entity_type, count in entity_types.items():
            print(f"{entity_type}: {count}")
    
    def create_cytoscape_viewer(self, output_file="cytoscape_graph_viewer_fixed.html"):
        """Create an advanced interactive viewer using Cytoscape.js with built-in layouts"""
        if not self.graph:
            print("No graph loaded")
            return
        
        # Prepare data for Cytoscape.js
        elements = []
        
        # Add nodes
        for node_id, data in self.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'default')
            description = data.get('description', 'No description available')
            degree = self.graph.degree[node_id]
            
            # Calculate node size based on description length and connections
            base_size = 40
            description_bonus = min(len(description) // 100, 30)  # Bonus for longer descriptions
            degree_bonus = degree * 8  # Bonus for more connections
            node_size = base_size + description_bonus + degree_bonus
            
            elements.append({
                'data': {
                    'id': node_id,
                    'label': node_id,
                    'entity_type': entity_type,
                    'description': description[:500] + "..." if len(description) > 500 else description,
                    'degree': degree,
                    'size': node_size,
                    'color': self.node_colors.get(entity_type, self.node_colors['default'])
                }
            })
        
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            edge_label = data.get('label', '')
            edge_description = data.get('description', '')
            
            elements.append({
                'data': {
                    'id': f"{source}-{target}",
                    'source': source,
                    'target': target,
                    'label': edge_label,
                    'description': edge_description
                }
            })
        
        # Create the HTML file with Cytoscape.js
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced GraphML Viewer - Cytoscape.js</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
        }}
        
        .container {{
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 15px 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            z-index: 1000;
        }}
        
        .title {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            text-align: center;
        }}
        
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
            justify-content: center;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .search-box {{
            padding: 10px 15px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 14px;
            width: 250px;
            background: white;
            transition: all 0.3s ease;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .btn-primary {{
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .btn-secondary {{
            background: #ecf0f1;
            color: #2c3e50;
        }}
        
        .btn-secondary:hover {{
            background: #d5dbdb;
        }}
        
        .stats {{
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        
        .legend {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 500;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .graph-container {{
            flex: 1;
            position: relative;
            background: white;
            margin: 0 10px 10px 10px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        #cy {{
            width: 100%;
            height: 100%;
        }}
        
        .info-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            max-width: 350px;
            max-height: 400px;
            overflow-y: auto;
            backdrop-filter: blur(10px);
            display: none;
            z-index: 1000;
        }}
        
        .info-panel h3 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 18px;
        }}
        
        .info-panel p {{
            color: #7f8c8d;
            line-height: 1.5;
            margin-bottom: 8px;
        }}
        
        .info-panel .close-btn {{
            position: absolute;
            top: 10px;
            right: 15px;
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: #95a5a6;
        }}
        
        .layout-selector {{
            background: white;
            border: 2px solid #ddd;
            border-radius: 20px;
            padding: 8px 12px;
            font-size: 14px;
        }}
        
        .floating-help {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(44, 62, 80, 0.9);
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-size: 12px;
            max-width: 300px;
            backdrop-filter: blur(10px);
        }}
        
        .floating-help h4 {{
            margin-bottom: 8px;
            color: #3498db;
        }}
        
        .floating-help ul {{
            list-style: none;
            margin: 0;
            padding: 0;
        }}
        
        .floating-help li {{
            margin-bottom: 4px;
            padding-left: 16px;
            position: relative;
        }}
        
        .floating-help li:before {{
            content: "•";
            color: #3498db;
            position: absolute;
            left: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🚀 Advanced Knowledge Graph Viewer</div>
            <div class="controls">
                <div class="control-group">
                    <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search nodes by name or description..." />
                </div>
                
                <div class="control-group">
                    <select class="layout-selector" id="layoutSelector">
                        <option value="cose">Physics Layout (CoSE)</option>
                        <option value="circle">Circle Layout</option>
                        <option value="grid">Grid Layout</option>
                        <option value="random">Random Layout</option>
                        <option value="concentric">Concentric Layout</option>
                        <option value="breadthfirst">Breadth First</option>
                    </select>
                </div>
                
                <button class="btn btn-primary" id="fitBtn">🎯 Fit View</button>
                <button class="btn btn-secondary" id="resetBtn">🔄 Reset</button>
                <button class="btn btn-secondary" id="exportBtn">💾 Export PNG</button>
                
                <div class="stats">
                    📊 {self.graph.number_of_nodes()} Nodes | 🔗 {self.graph.number_of_edges()} Edges
                </div>
                
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {self.node_colors['organization']}"></div>
                        <span>Organizations</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {self.node_colors['person']}"></div>
                        <span>People</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {self.node_colors['category']}"></div>
                        <span>Categories</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {self.node_colors['event']}"></div>
                        <span>Events</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="graph-container">
            <div id="cy"></div>
            
            <div class="info-panel" id="infoPanel">
                <button class="close-btn" id="closeInfo">&times;</button>
                <div id="infoContent"></div>
            </div>
            
            <div class="floating-help">
                <h4>💡 Navigation Tips</h4>
                <ul>
                    <li>Click nodes to see details</li>
                    <li>Drag to pan around</li>
                    <li>Mouse wheel to zoom</li>
                    <li>Right-click node to highlight neighbors</li>
                    <li>Double-click background to deselect</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        // Graph data
        const elements = {json.dumps(elements, indent=2)};
        
        // Initialize Cytoscape
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            
            elements: elements,
            
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'background-color': 'data(color)',
                        'label': 'data(label)',
                        'width': 'data(size)',
                        'height': 'data(size)',
                        'font-size': '12px',
                        'font-weight': 'bold',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'color': '#2c3e50',
                        'text-outline-width': 2,
                        'text-outline-color': '#ffffff',
                        'border-width': 3,
                        'border-color': '#ffffff',
                        'transition-property': 'background-color, border-color, width, height',
                        'transition-duration': '0.3s'
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-color': '#e74c3c',
                        'border-width': 5
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 3,
                        'line-color': '#bdc3c7',
                        'target-arrow-color': '#bdc3c7',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.7,
                        'transition-property': 'line-color, width, opacity',
                        'transition-duration': '0.3s'
                    }}
                }},
                {{
                    selector: '.highlighted',
                    style: {{
                        'background-color': '#f39c12',
                        'line-color': '#f39c12',
                        'target-arrow-color': '#f39c12',
                        'transition-property': 'background-color, line-color, target-arrow-color',
                        'transition-duration': '0.5s'
                    }}
                }},
                {{
                    selector: '.faded',
                    style: {{
                        'opacity': 0.25,
                        'text-opacity': 0.25
                    }}
                }}
            ],
            
            layout: {{
                name: 'cose',
                animate: true,
                animationDuration: 1000,
                fit: true,
                padding: 50,
                nodeRepulsion: function( node ){{ return 2048; }},
                nodeOverlap: 4,
                idealEdgeLength: function( edge ){{ return 32; }},
                edgeElasticity: function( edge ){{ return 32; }},
                nestingFactor: 1.2,
                gravity: 1,
                numIter: 1000,
                initialTemp: 1000,
                coolingFactor: 0.99,
                minTemp: 1.0
            }},
            
            wheelSensitivity: 0.5,
            minZoom: 0.1,
            maxZoom: 5
        }});
        
        // Event handlers
        const infoPanel = document.getElementById('infoPanel');
        const infoContent = document.getElementById('infoContent');
        const searchBox = document.getElementById('searchBox');
        const layoutSelector = document.getElementById('layoutSelector');
        
        // Node click handler
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            
            infoContent.innerHTML = `
                <h3>${{data.label}}</h3>
                <p><strong>Type:</strong> ${{data.entity_type}}</p>
                <p><strong>Connections:</strong> ${{data.degree}}</p>
                <p><strong>Description:</strong></p>
                <p style="font-style: italic; max-height: 200px; overflow-y: auto;">${{data.description}}</p>
            `;
            
            infoPanel.style.display = 'block';
        }});
        
        // Background click handler
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                infoPanel.style.display = 'none';
                cy.elements().removeClass('highlighted faded');
            }}
        }});
        
        // Right-click to highlight neighbors
        cy.on('cxttap', 'node', function(evt) {{
            const node = evt.target;
            const neighbors = node.neighborhood().add(node);
            
            cy.elements().addClass('faded');
            neighbors.removeClass('faded').addClass('highlighted');
        }});
        
        // Close info panel
        document.getElementById('closeInfo').onclick = function() {{
            infoPanel.style.display = 'none';
        }};
        
        // Search functionality
        let searchTimeout;
        searchBox.addEventListener('input', function() {{
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {{
                const searchTerm = this.value.toLowerCase();
                
                if (searchTerm === '') {{
                    cy.elements().removeClass('highlighted faded');
                    return;
                }}
                
                const matchingNodes = cy.nodes().filter(function(node) {{
                    const data = node.data();
                    return data.label.toLowerCase().includes(searchTerm) ||
                           data.description.toLowerCase().includes(searchTerm) ||
                           data.entity_type.toLowerCase().includes(searchTerm);
                }});
                
                if (matchingNodes.length > 0) {{
                    cy.elements().addClass('faded');
                    matchingNodes.removeClass('faded').addClass('highlighted');
                    
                    // Fit to matching nodes
                    cy.fit(matchingNodes, 100);
                }}
            }}, 300);
        }});
        
        // Layout selector
        layoutSelector.addEventListener('change', function() {{
            const layoutName = this.value;
            const layoutOptions = {{
                cose: {{
                    name: 'cose',
                    animate: true,
                    animationDuration: 1000,
                    fit: true,
                    padding: 50,
                    nodeRepulsion: function( node ){{ return 2048; }},
                    nodeOverlap: 4,
                    idealEdgeLength: function( edge ){{ return 32; }},
                    edgeElasticity: function( edge ){{ return 32; }},
                    nestingFactor: 1.2,
                    gravity: 1,
                    numIter: 1000
                }},
                circle: {{
                    name: 'circle',
                    animate: true,
                    fit: true,
                    padding: 50
                }},
                grid: {{
                    name: 'grid',
                    animate: true,
                    fit: true,
                    padding: 50,
                    rows: Math.ceil(Math.sqrt(cy.nodes().length))
                }},
                random: {{
                    name: 'random',
                    animate: true,
                    fit: true,
                    padding: 50
                }},
                concentric: {{
                    name: 'concentric',
                    animate: true,
                    fit: true,
                    padding: 50,
                    concentric: function( node ){{
                        return node.degree();
                    }},
                    levelWidth: function( nodes ){{
                        return 2;
                    }}
                }},
                breadthfirst: {{
                    name: 'breadthfirst',
                    animate: true,
                    fit: true,
                    padding: 50,
                    directed: false,
                    roots: cy.nodes().first(),
                    spacingFactor: 1.5
                }}
            }};
            
            cy.layout(layoutOptions[layoutName]).run();
        }});
        
        // Control buttons
        document.getElementById('fitBtn').onclick = function() {{
            cy.fit(undefined, 50);
        }};
        
        document.getElementById('resetBtn').onclick = function() {{
            cy.elements().removeClass('highlighted faded');
            searchBox.value = '';
            infoPanel.style.display = 'none';
            cy.fit(undefined, 50);
        }};
        
        document.getElementById('exportBtn').onclick = function() {{
            const png = cy.png({{
                output: 'blob',
                bg: 'white',
                full: true,
                scale: 2
            }});
            
            const link = document.createElement('a');
            link.download = 'knowledge_graph.png';
            link.href = URL.createObjectURL(png);
            link.click();
        }};
        
        // Initial fit
        cy.ready(function() {{
            cy.fit(undefined, 50);
        }});
        
        // Responsive handling
        window.addEventListener('resize', function() {{
            cy.resize();
            cy.fit(undefined, 50);
        }});
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Fixed Cytoscape.js viewer saved as: {output_file}")
        return output_file

def main():
    """Main function to run the Cytoscape GraphML viewer"""
    print("🚀 Advanced GraphML Viewer - Powered by Cytoscape.js (Fixed Version)")
    print("=" * 70)
    
    viewer = CytoscapeGraphMLViewer()
    
    # Load default GraphML file if it exists, otherwise prompt user
    default_file = "./data/vespa_emails_rag1/graph_chunk_entity_relation.graphml"
    if os.path.exists(default_file):
        print(f"📂 Loading default GraphML file: {default_file}")
        success = viewer.load_graphml(default_file)
    else:
        print("📂 Please select a GraphML file to visualize...")
        success = viewer.load_graphml()
    
    if not success:
        print("❌ Failed to load GraphML file. Exiting.")
        return
    
    # Analyze the graph
    viewer.analyze_graph()
    
    print("\n🎨 Creating advanced visualization...")
    
    # Create Cytoscape.js viewer
    html_file = viewer.create_cytoscape_viewer(output_file="cytoscape_graph_viewer_smart_route.html")
    
    print(f"\n✅ Generated file:")
    print(f"🌟 Fixed Cytoscape.js Viewer: {html_file}")
    print(f"\n🎯 Features included:")
    print(f"   • Built-in layout algorithms (CoSE, Circle, Grid, Random, Concentric, Breadth-First)")
    print(f"   • Real-time search and filtering")
    print(f"   • Interactive node information panels")
    print(f"   • Neighbor highlighting")
    print(f"   • Smooth animations and transitions")
    print(f"   • PNG export functionality")
    print(f"   • Responsive design")
    print(f"   • Professional styling")
    
    # Ask user if they want to open the file
    root = tk.Tk()
    root.withdraw()
    
    if messagebox.askyesno("Open Viewer", "Would you like to open the fixed graph viewer in your browser?"):
        webbrowser.open(f"file://{os.path.abspath(html_file)}")
    
    root.destroy()
    print("\n🎉 Done! Enjoy exploring your knowledge graph!")

if __name__ == "__main__":
    main()
