"""manifest 声明式注册表完整性测试。"""
from septmuse.services.registry import _DEFAULTS, BACKEND_MANIFEST, BackendEntry

CAPABILITIES = ["vector_store", "embedder", "llm", "reranker",
                "entity_extractor", "keyword_index", "graph_store", "search_recipe"]


def test_all_capabilities_present():
    for cap in CAPABILITIES:
        assert cap in BACKEND_MANIFEST, f"能力 {cap} 不在 manifest"


def test_each_capability_has_default():
    for cap in CAPABILITIES:
        assert cap in _DEFAULTS, f"能力 {cap} 无代码默认"
        default = _DEFAULTS[cap]
        assert default == "" or default in BACKEND_MANIFEST[cap], \
            f"能力 {cap} 的默认 {default} 不在 manifest"


def test_zero_dep_backends_exist():
    zero_dep = []
    for cap, backends in BACKEND_MANIFEST.items():
        for name, entry in backends.items():
            if entry.deps == ():
                zero_dep.append((cap, name))
    for cap in CAPABILITIES:
        default = _DEFAULTS[cap]
        if default:
            entry = BACKEND_MANIFEST[cap][default]
            assert entry.deps == (), f"能力 {cap} 默认 {default} 有依赖，破坏零配置"


def test_backend_entry_fields():
    entry = BACKEND_MANIFEST["vector_store"]["sqlite"]
    assert isinstance(entry, BackendEntry)
    assert entry.module.startswith("septmuse.")
    assert isinstance(entry.cls, str)
    assert entry.config_cls is not None or entry.cls == "get_recipe"
    assert isinstance(entry.deps, tuple)


def test_search_recipe_uses_function():
    for _name, entry in BACKEND_MANIFEST["search_recipe"].items():
        assert entry.cls == "get_recipe"
        assert entry.config_cls is None
