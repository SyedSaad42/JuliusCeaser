import pytest

import ask


@pytest.fixture(scope="module")
def H():
    return ask.build()


def test_clean_env_strips_quotes_and_spaces(monkeypatch):
    monkeypatch.setenv("T_KEY", ' "abc" ')
    monkeypatch.setenv("T_EMPTY", "  ")
    assert ask.clean_env("T_KEY") == "abc"
    assert ask.clean_env("T_EMPTY") is None and ask.clean_env("T_MISSING") is None


def test_credit_error_detection():
    class E(Exception):
        def __init__(self, msg, code=400):
            super().__init__(msg)
            self.status_code = code

    assert ask.is_credit_error(E("Your credit balance is too low to access the Anthropic API"))
    assert ask.is_credit_error(E("x", 402))
    assert not ask.is_credit_error(E("invalid model"))


def test_leak_guard_catches_the_future_but_not_the_past():
    for bad in ("Brutus killed Caesar", "after his assassination", "on the Ides of March", "the battle of Philippi"):
        assert ask.LEAK.search(bad), bad
    for ok in ("Brutus's father was killed in 77 BCE on Pompey's orders", "Caesar pardoned Cassius"):
        assert not ask.LEAK.search(ok), ok


def test_find_node_and_edges(H):
    assert "marcus_brutus" in ask.tool_find_node(H, "Brutus")
    assert "no node matches" in ask.tool_find_node(H, "zzzzqq")
    assert "unknown node" in ask.tool_get_edges(H, "nobody")
    assert "unknown edge" in ask.tool_get_edge(H, "nothing")


def test_connection_finds_a_chain(H):
    out = ask.tool_connection(H, "cassius", "mark_antony")
    assert "hop" in out
    assert ask.examined_edges(H, "connection", {}, out)             # the walked edges are reported


def test_policy_touches_finds_veterans_and_land(H):
    out = ask.tool_policy_touches(H, "land settlement for veterans")
    assert "MOTIVES" in out and "caesars_veterans_and_army" in out
    assert "nothing in the graph" in ask.tool_policy_touches(H, "xyzzy")
    assert ask.examined_edges(H, "policy_touches", {}, out)


def test_tool_errors_go_back_to_the_model(H):
    assert "tool error" in ask.run_tool(H, "get_edges", {"bad_argument": 1})
