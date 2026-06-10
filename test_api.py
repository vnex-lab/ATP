"""Comprehensive test suite for VnexAI API endpoints."""
import asyncio
import json
import sys
from io import BytesIO
from typing import Dict, Any, List

# Test data
VALID_JSON_DATA = [
    {"user": "What is Python?", "bot": "Python is a programming language."},
    {"user": "How to learn Python?", "bot": "Read documentation, practice coding, join communities."},
]

VALID_CSV_DATA = """user,bot
What is AI?,Artificial Intelligence is...
How does it work?,It works by...
"""

VALID_JSONL_DATA = '{"user": "What is ML?", "bot": "Machine Learning is a subset of AI."}\n{"user": "Why ML?", "bot": "ML enables automated learning from data."}\n'

VALID_TSV_DATA = """user\tbot
Hello bot\tHello human
How are you?\tI'm doing well
"""

def test_parse_json():
    """Test JSON file parsing."""
    from api import parse_uploaded_file
    content = json.dumps(VALID_JSON_DATA).encode('utf-8')
    result = parse_uploaded_file(content, 'test.json')
    assert len(result) == 2, f"Expected 2 pairs, got {len(result)}"
    assert result[0]['user'] == "What is Python?"
    print("[PASS] JSON parsing test passed")


def test_parse_jsonl():
    """Test JSONL file parsing."""
    from api import parse_uploaded_file
    content = VALID_JSONL_DATA.encode('utf-8')
    result = parse_uploaded_file(content, 'test.jsonl')
    assert len(result) == 2, f"Expected 2 pairs, got {len(result)}"
    print("[PASS] JSONL parsing test passed")


def test_parse_csv():
    """Test CSV file parsing."""
    from api import parse_uploaded_file
    content = VALID_CSV_DATA.encode('utf-8')
    result = parse_uploaded_file(content, 'test.csv')
    assert len(result) == 2, f"Expected 2 pairs, got {len(result)}"
    assert result[0]['user'] == "What is AI?"
    print("[PASS] CSV parsing test passed")


def test_parse_tsv():
    """Test TSV file parsing."""
    from api import parse_uploaded_file
    content = VALID_TSV_DATA.encode('utf-8')
    result = parse_uploaded_file(content, 'test.tsv')
    assert len(result) == 2, f"Expected 2 pairs, got {len(result)}"
    print("[PASS] TSV parsing test passed")


def test_parse_txt_with_separators():
    """Test TXT file parsing with separators."""
    from api import parse_uploaded_file
    txt_data = "Q: What is Python? | A: Python is a language\nQ: How to code? | A: By practicing"
    content = txt_data.encode('utf-8')
    result = parse_uploaded_file(content, 'test.txt')
    assert len(result) >= 1, f"Expected at least 1 pair, got {len(result)}"
    print("[PASS] TXT parsing test passed")


def test_parse_invalid_json():
    """Test handling of invalid JSON."""
    from api import parse_uploaded_file
    content = b"not valid json {{{ ]]]"
    result = parse_uploaded_file(content, 'test.json')
    assert isinstance(result, list), "Should return a list even for invalid JSON"
    print("[PASS] Invalid JSON handling test passed")


def test_parse_encoding_fallback():
    """Test encoding fallback for non-UTF8 files."""
    from api import parse_uploaded_file
    # Create latin-1 encoded data
    csv_data = "user,bot\nHello,World"
    content = csv_data.encode('latin-1')
    result = parse_uploaded_file(content, 'test.csv')
    # Should handle gracefully
    assert isinstance(result, list), "Should return a list"
    print("[PASS] Encoding fallback test passed")


def test_plugin_manager():
    """Test plugin loading."""
    from plugin_manager import plugin_info, load_plugins, load_mods
    
    info = plugin_info()
    assert 'plugins' in info, "Should have plugins key"
    assert 'mods' in info, "Should have mods key"
    assert 'errors' in info, "Should have errors key"
    print("[PASS] Plugin manager test passed")


def test_plugin_loading_and_reloading():
    """Test plugin loading, status, and reloading."""
    from plugin_manager import get_plugin_status, reload_extensions
    
    status = get_plugin_status()
    assert 'plugins' in status, "Should have plugins in status"
    assert 'mods' in status, "Should have mods in status"
    
    reload_result = reload_extensions()
    assert 'plugins_loaded' in reload_result, "Should have plugins_loaded in reload result"
    assert 'mods_loaded' in reload_result, "Should have mods_loaded in reload result"
    assert reload_result['plugins_loaded'] >= 0, "Should have non-negative plugins loaded"
    print("[PASS] Plugin loading and reloading test passed")


def test_tokenizer_initialization():
    """Test tokenizer initialization."""
    from chatbot_tokenizer import ChatbotTokenizer
    
    vocab = ["hello", "world", "test"]
    tokenizer = ChatbotTokenizer(vocab)
    
    # Test encoding
    tokens = tokenizer.encode("hello world", add_special_tokens=True)
    assert len(tokens) > 0, "Should encode to tokens"
    
    # Test decoding
    text = tokenizer.decode(tokens)
    assert isinstance(text, str), "Should decode to string"
    print("[PASS] Tokenizer initialization test passed")


def test_model_initialization():
    """Test model initialization."""
    from transformer_model import TransformerChatbot
    
    model = TransformerChatbot(
        vocab_size=100,
        embed_dim=64,
        num_heads=2,
        num_layers=1,
        ff_dim=128,
        max_seq_len=128,
    )
    assert model is not None, "Should initialize model"
    print("[PASS] Model initialization test passed")


def test_parse_code_debug_format():
    """Test parsing code debugging format."""
    from api import parse_uploaded_file
    
    code_debug_data = [
        {
            "problem_id": "p001",
            "language": "Python",
            "original_status": "Runtime Error",
            "original_src": "print(1/0)",
            "changed_src": "print(1) # Fixed"
        },
        {
            "problem_id": "p002",
            "language": "Python",
            "original_status": "Wrong Answer",
            "original_src": "return x + y",
            "changed_src": "return x * y"
        }
    ]
    
    content = json.dumps(code_debug_data).encode('utf-8')
    result = parse_uploaded_file(content, 'debug.json')
    
    assert len(result) == 2, f"Expected 2 pairs, got {len(result)}"
    assert "Fix this Python code" in result[0]['user']
    print("[PASS] Code debug format parsing test passed")


def test_api_startup():
    """Test API startup without errors."""
    try:
        from api import app
        assert app is not None, "Should load API app"
        print("[PASS] API startup test passed")
    except Exception as e:
        print(f"[FAIL] API startup test failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*50)
    print("Running VnexAI Comprehensive Test Suite")
    print("="*50 + "\n")
    
    tests = [
        test_parse_json,
        test_parse_jsonl,
        test_parse_csv,
        test_parse_tsv,
        test_parse_txt_with_separators,
        test_parse_invalid_json,
        test_parse_encoding_fallback,
        test_parse_code_debug_format,
        test_plugin_manager,
        test_plugin_loading_and_reloading,
        test_tokenizer_initialization,
        test_model_initialization,
        test_api_startup,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} errored: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("="*50 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
