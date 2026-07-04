from circuit_tracer.utils.caching import _config_requests_exact_chunked_provider


def test_cache_exact_provider_request_honors_capability_metadata():
    assert _config_requests_exact_chunked_provider(
        {
            "transcoder_capabilities": {
                "supports_exact_chunked_provider": True,
            }
        }
    )


def test_cache_exact_provider_request_honors_fingerprint_metadata():
    assert _config_requests_exact_chunked_provider(
        {
            "transcoder_provider_fingerprint": {
                "supports_exact_chunked_provider": True,
            }
        }
    )


def test_cache_exact_provider_request_prefers_explicit_legacy_flags():
    assert not _config_requests_exact_chunked_provider(
        {
            "supports_exact_chunked_provider": False,
            "transcoder_capabilities": {
                "supports_exact_chunked_provider": True,
            },
        }
    )
