from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment='The Round Table')

# Add nodes and edges to represent the structure of the code
dot.node('A', 'main')
dot.node('B', 'load_data')
dot.node('C', 'split')
dot.node('D', 'plot_metrics')
dot.node('E', 'Classifier Selection')
dot.node('F', 'SVM Settings')
dot.node('G', 'Logistic Regression Settings')
dot.node('H', 'Random Forest Settings')
dot.node('I', 'Show Raw Data')

dot.edges(['AB', 'AC', 'AD', 'AE'])
dot.edge('E', 'F', label='SVM')
dot.edge('E', 'G', label='Logistic Regression')
dot.edge('E', 'H', label='Random Forest')
dot.edge('E', 'I', label='Raw Data')

# Viewing the graph
dot.view()
