# causal_graph/explainer.py
# Optional: Visualize or explain the causal graph and path results

import matplotlib.pyplot as plt

class CausalGraphExplainer:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def plot_graph(self, highlight: Optional[List[Tuple[str, str]]] = None):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue')
        nx.draw_networkx_labels(self.graph, pos)

        if highlight:
            edge_colors = ["red" if (u, v) in highlight else "gray" for u, v in self.graph.edges()]
        else:
            edge_colors = "gray"

        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=2)
        plt.title("Causal Graph View")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def print_paths(self, nodes: List[str]):
        for src in nodes:
            for tgt in nodes:
                if src != tgt and nx.has_path(self.graph, src, tgt):
                    print(f"{src} → {tgt}:", " → ".join(nx.shortest_path(self.graph, src, tgt)))