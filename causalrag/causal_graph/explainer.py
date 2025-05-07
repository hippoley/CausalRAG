# causal_graph/explainer.py
# Visualize or explain the causal graph and path results

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set, Any, Union
import logging
import io
import base64
from PIL import Image
import textwrap

logger = logging.getLogger(__name__)

class CausalGraphExplainer:
    """Explain and visualize causal graphs and query results"""
    
    def __init__(self, graph: nx.DiGraph, node_text: Optional[Dict[str, str]] = None):
        """
        Initialize the causal graph explainer
        
        Args:
            graph: NetworkX DiGraph containing the causal graph
            node_text: Optional mapping from node IDs to display text
        """
        self.graph = graph
        self.node_text = node_text or {}
    
    def plot_graph(self, 
                  highlight_nodes: Optional[List[str]] = None,
                  highlight_edges: Optional[List[Tuple[str, str]]] = None,
                  title: str = "Causal Graph",
                  figsize: Tuple[int, int] = (12, 8),
                  save_path: Optional[str] = None) -> Optional[str]:
        """
        Visualize the causal graph with optional highlighting
        
        Args:
            highlight_nodes: List of node IDs to highlight
            highlight_edges: List of edge tuples to highlight
            title: Plot title
            figsize: Figure size as (width, height)
            save_path: Optional path to save the visualization
            
        Returns:
            If save_path is None, returns a base64 encoded image
        """
        try:
            # Create figure
            plt.figure(figsize=figsize)
            
            # Use spring layout for node positioning
            pos = nx.spring_layout(self.graph, seed=42)
            
            # Prepare node labels (use node_text if available)
            node_labels = {}
            for node in self.graph.nodes():
                # Use node_text if available, otherwise use the node ID
                # Wrap long text
                text = self.node_text.get(node, str(node))
                node_labels[node] = '\n'.join(textwrap.wrap(text, 15))
            
            # Set node colors
            node_colors = []
            for node in self.graph.nodes():
                if highlight_nodes and node in highlight_nodes:
                    node_colors.append("lightcoral")
                else:
                    node_colors.append("lightblue")
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph, pos, 
                node_color=node_colors,
                node_size=2000,
                alpha=0.8
            )
            
            # Draw node labels
            nx.draw_networkx_labels(
                self.graph, pos, 
                labels=node_labels,
                font_size=9
            )
            
            # Draw edges with different colors for highlighted edges
            edge_colors = []
            edge_widths = []
            
            for u, v in self.graph.edges():
                if highlight_edges and (u, v) in highlight_edges:
                    edge_colors.append("red")
                    edge_widths.append(2.5)
                else:
                    edge_colors.append("gray")
                    edge_widths.append(1.0)
            
            # Draw edges
            nx.draw_networkx_edges(
                self.graph, pos, 
                edge_color=edge_colors,
                width=edge_widths,
                arrowsize=20,
                connectionstyle='arc3,rad=0.1'  # Curved edges
            )
            
            # Add edge weights as labels if available
            edge_labels = {}
            for u, v, data in self.graph.edges(data=True):
                if 'weight' in data:
                    edge_labels[(u, v)] = f"{data['weight']:.2f}"
            
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    self.graph, pos,
                    edge_labels=edge_labels,
                    font_size=8
                )
            
            # Set title and remove axes
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            
            # Save to file or return as base64
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                # Convert to base64 for embedding
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                return img_str
                
        except Exception as e:
            logger.error(f"Error plotting graph: {e}")
            plt.close()
            return None
    
    def print_paths(self, 
                   nodes: List[str], 
                   max_path_length: int = 5,
                   include_weights: bool = True) -> str:
        """
        Generate a text explanation of paths between nodes
        
        Args:
            nodes: List of node IDs to find paths between
            max_path_length: Maximum path length to consider
            include_weights: Whether to include edge weights
            
        Returns:
            Text explanation of paths
        """
        if not nodes or len(nodes) < 2:
            return "Insufficient nodes to find paths."
        
        explanation = []
        path_count = 0
        
        for i, src in enumerate(nodes):
            src_name = self.node_text.get(src, src)
            
            for tgt in nodes[i+1:]:
                if src == tgt:
                    continue
                    
                tgt_name = self.node_text.get(tgt, tgt)
                
                try:
                    # Try both directions
                    for direction, path_type in [
                        ((src, tgt), f"Effects of {src_name} on {tgt_name}"),
                        ((tgt, src), f"Causes of {tgt_name} from {src_name}")
                    ]:
                        start, end = direction
                        
                        if nx.has_path(self.graph, start, end):
                            # Find all simple paths within length limit
                            simple_paths = list(nx.all_simple_paths(
                                self.graph, start, end, cutoff=max_path_length
                            ))
                            
                            if simple_paths:
                                explanation.append(f"\n{path_type}:")
                                
                                for j, path in enumerate(simple_paths[:3]):  # Limit to 3 paths
                                    path_str = []
                                    
                                    for k in range(len(path) - 1):
                                        node1 = self.node_text.get(path[k], path[k])
                                        node2 = self.node_text.get(path[k+1], path[k+1])
                                        
                                        if include_weights and 'weight' in self.graph[path[k]][path[k+1]]:
                                            weight = self.graph[path[k]][path[k+1]]['weight']
                                            path_str.append(f"{node1} --({weight:.2f})--> {node2}")
                                        else:
                                            path_str.append(f"{node1} --> {node2}")
                                    
                                    path_full = " ".join(path_str)
                                    explanation.append(f"  Path {j+1}: {path_full}")
                                    path_count += 1
                
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    continue
        
        if path_count == 0:
            return "No paths found between the specified nodes."
        
        return "\n".join(explanation)
    
    def summarize_graph(self) -> str:
        """
        Generate a summary of the causal graph
        
        Returns:
            Text summary of the graph
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return "Empty causal graph (no nodes or relationships)."
        
        # Basic statistics
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        summary = [
            f"Causal Graph Summary:",
            f"- Concepts: {num_nodes}",
            f"- Relationships: {num_edges}"
        ]
        
        # Most central nodes (by degree)
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        
        # Top causes (nodes with high out-degree)
        top_causes = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_causes:
            summary.append("\nTop causes (most outgoing relationships):")
            for node, degree in top_causes:
                if degree > 0:
                    node_name = self.node_text.get(node, node)
                    summary.append(f"- {node_name} ({degree} effects)")
        
        # Top effects (nodes with high in-degree)
        top_effects = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_effects:
            summary.append("\nTop effects (most incoming relationships):")
            for node, degree in top_effects:
                if degree > 0:
                    node_name = self.node_text.get(node, node)
                    summary.append(f"- {node_name} ({degree} causes)")
        
        # Try to identify clusters or components
        try:
            # Get weakly connected components
            components = list(nx.weakly_connected_components(self.graph))
            if len(components) > 1:
                summary.append(f"\nThe graph contains {len(components)} disconnected causal networks.")
                
                # List the largest components
                largest_components = sorted(components, key=len, reverse=True)[:3]
                for i, component in enumerate(largest_components):
                    if len(component) > 1:
                        sample_nodes = [self.node_text.get(n, n) for n in list(component)[:3]]
                        summary.append(f"- Network {i+1}: {len(component)} concepts including {', '.join(sample_nodes)}...")
        except:
            pass
        
        return "\n".join(summary)
    
    def explain_query_relevance(self, 
                               query: str, 
                               retriever: 'CausalPathRetriever') -> str:
        """
        Explain why certain nodes/paths are relevant to a query
        
        Args:
            query: User query
            retriever: CausalPathRetriever instance
            
        Returns:
            Text explanation of query relevance
        """
        # Get relevant nodes
        relevant_nodes = retriever.retrieve_nodes(query, top_k=5)
        
        if not relevant_nodes:
            return f"No concepts directly relevant to '{query}' were found in the causal graph."
        
        explanation = [f"Query: '{query}'\n"]
        explanation.append("Directly relevant concepts:")
        
        # List relevant nodes with scores
        for node_id, score in relevant_nodes:
            node_name = self.node_text.get(node_id, node_id)
            explanation.append(f"- {node_name} (relevance: {score:.2f})")
        
        # Get paths between relevant nodes
        paths = retriever.retrieve_paths(query, max_paths=3)
        
        if paths:
            explanation.append("\nRelevant causal pathways:")
            for i, path in enumerate(paths):
                explanation.append(f"{i+1}. {' → '.join(path)}")
                
            # Add interpretation
            explanation.append("\nInterpretation:")
            explanation.append(f"These pathways show how concepts related to '{query}' influence each other through causal mechanisms.")
        
        return "\n".join(explanation)
    
    def generate_graph_viz_html(self,
                               highlight_nodes: Optional[List[str]] = None,
                               highlight_edges: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Generate a standalone HTML file with an interactive graph visualization
        using D3.js for better interactivity
        
        Args:
            highlight_nodes: Optional list of nodes to highlight
            highlight_edges: Optional list of edges to highlight
            
        Returns:
            HTML string for interactive visualization
        """
        # Convert graph to JSON format for D3
        nodes = []
        for node in self.graph.nodes():
            nodes.append({
                "id": node,
                "name": self.node_text.get(node, node),
                "highlighted": highlight_nodes and node in highlight_nodes
            })
        
        links = []
        for u, v, data in self.graph.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0),
                "highlighted": highlight_edges and (u, v) in highlight_edges
            })
        
        # Create HTML with embedded D3.js
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Interactive Causal Graph</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; }
                #graph-container { width: 100%; height: 800px; }
                .node { stroke: #fff; stroke-width: 1.5px; }
                .node.highlighted { stroke: #ff0000; stroke-width: 2px; }
                .link { stroke: #999; stroke-opacity: 0.6; }
                .link.highlighted { stroke: #ff0000; stroke-width: 2px; }
                .node-label { font-size: 10px; pointer-events: none; }
                .controls { padding: 10px; background: #f8f8f8; }
            </style>
        </head>
        <body>
            <div class="controls">
                <button id="zoom-in">+</button>
                <button id="zoom-out">-</button>
                <button id="reset">Reset</button>
                <input type="checkbox" id="show-labels" checked> <label for="show-labels">Show Labels</label>
            </div>
            <div id="graph-container"></div>
            
            <script>
            // Graph data
            const graph = {
                "nodes": %s,
                "links": %s
            };
            
            // Create the visualization
            const width = document.getElementById('graph-container').clientWidth;
            const height = document.getElementById('graph-container').clientHeight;
            
            const svg = d3.select("#graph-container").append("svg")
                .attr("width", width)
                .attr("height", height);
                
            const g = svg.append("g");
            
            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on("zoom", (event) => {
                    g.attr("transform", event.transform);
                });
                
            svg.call(zoom);
            
            // Create force simulation
            const simulation = d3.forceSimulation(graph.nodes)
                .force("link", d3.forceLink(graph.links).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-200))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("x", d3.forceX(width / 2).strength(0.05))
                .force("y", d3.forceY(height / 2).strength(0.05));
            
            // Draw links
            const link = g.append("g")
                .selectAll("line")
                .data(graph.links)
                .enter().append("line")
                .attr("class", d => "link" + (d.highlighted ? " highlighted" : ""))
                .attr("stroke-width", d => Math.sqrt(d.weight) * 1.5);
                
            // Add arrowheads
            svg.append("defs").selectAll("marker")
                .data(["end"])
                .enter().append("marker")
                .attr("id", d => d)
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 25)
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5");
                
            link.attr("marker-end", "url(#end)");
            
            // Draw nodes
            const node = g.append("g")
                .selectAll("circle")
                .data(graph.nodes)
                .enter().append("circle")
                .attr("class", d => "node" + (d.highlighted ? " highlighted" : ""))
                .attr("r", 10)
                .attr("fill", d => d.highlighted ? "#ff9999" : "#69b3a2")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
                    
            // Add node labels
            const label = g.append("g")
                .selectAll("text")
                .data(graph.nodes)
                .enter().append("text")
                .attr("class", "node-label")
                .attr("text-anchor", "middle")
                .attr("dy", 3)
                .text(d => {
                    // Truncate long labels
                    return d.name.length > 20 ? d.name.substring(0, 17) + "..." : d.name;
                });
                
            // Add tooltips
            node.append("title")
                .text(d => d.name);
                
            // Update positions on simulation tick
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                    
                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y - 15);
            });
            
            // Drag functions
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
            
            // Controls
            document.getElementById("zoom-in").addEventListener("click", () => {
                svg.transition().call(zoom.scaleBy, 1.3);
            });
            
            document.getElementById("zoom-out").addEventListener("click", () => {
                svg.transition().call(zoom.scaleBy, 0.7);
            });
            
            document.getElementById("reset").addEventListener("click", () => {
                svg.transition().call(zoom.transform, d3.zoomIdentity);
            });
            
            document.getElementById("show-labels").addEventListener("change", function() {
                if (this.checked) {
                    label.style("display", "block");
                } else {
                    label.style("display", "none");
                }
            });
            </script>
        </body>
        </html>
        """ % (
            json.dumps(nodes, ensure_ascii=False),
            json.dumps(links, ensure_ascii=False)
        )
        
        return html