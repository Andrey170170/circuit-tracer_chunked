from circuit_tracer.frontend.graph_models import Node


def test_error_nodes_have_unique_parseable_python_and_javascript_ids() -> None:
    error_nodes = [Node.error_node(layer=1, pos=pos) for pos in range(3)]
    feature_nodes = [
        Node.feature_node(layer=1, pos=pos, feat_idx=pos) for pos in range(3)
    ]
    nodes = [*error_nodes, *feature_nodes]

    assert len({node.node_id for node in nodes}) == len(nodes)
    assert len({node.jsNodeId for node in nodes}) == len(nodes)
    assert [tuple(map(int, node.node_id.split("_"))) for node in error_nodes] == [
        (1, -1, 0),
        (1, -1, 1),
        (1, -1, 2),
    ]
