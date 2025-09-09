#!/usr/bin/env python3
"""
Test script to verify that all configurations match the provided examples exactly.
"""

from config import get_config, print_config_summary

def test_config_values():
    """Test that all critical configuration values match the provided examples."""
    config = get_config()

    print("🧪 Testing Configuration Values...")
    print("=" * 60)

    # Test critical consistency parameters
    critical_params = {
        "beam_size": 5,
        "temperature": 0,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "device": "cpu",
        "compute_type": "int8",
        "quant": "8bit",
        "batch_size": 8
    }

    all_passed = True
    for param, expected_value in critical_params.items():
        actual_value = getattr(config, param)
        status = "✅ PASS" if actual_value == expected_value else "❌ FAIL"
        print(f"{status} {param}: {actual_value} (expected: {expected_value})")
        if actual_value != expected_value:
            all_passed = False

    print("\n" + "=" * 60)

    # Test model-specific kwargs
    print("\n🔧 Testing Model-Specific Kwargs...")

    # OpenAI Whisper kwargs
    openai_kwargs = config.get_openai_whisper_kwargs()
    expected_openai = {
        "beam_size": 5,
        "temperature": 0,
        "word_timestamps": True,
        "condition_on_previous_text": False
    }

    print("\n📋 OpenAI Whisper kwargs:")
    for key, expected in expected_openai.items():
        actual = openai_kwargs.get(key, "MISSING")
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {key}: {actual}")

    # faster-whisper kwargs
    faster_kwargs = config.get_faster_whisper_kwargs()
    expected_faster = {
        "beam_size": 5,
        "temperature": 0,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "vad_filter": False
    }

    print("\n📋 faster-whisper kwargs:")
    for key, expected in expected_faster.items():
        actual = faster_kwargs.get(key, "MISSING")
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {key}: {actual}")

    # MLX Whisper kwargs
    mlx_kwargs = config.get_mlx_whisper_kwargs()
    expected_mlx = {
        "word_timestamps": True,
        "beam_size": 5,
        "temperature": 0,
        "initial_prompt": None,
        "condition_on_previous_text": False
    }

    print("\n📋 MLX Whisper kwargs:")
    for key, expected in expected_mlx.items():
        actual = mlx_kwargs.get(key, "MISSING")
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {key}: {actual}")

    # Lightning Whisper MLX kwargs
    lightning_kwargs = config.get_lightning_whisper_mlx_kwargs()
    expected_lightning = {
        "beam_size": 5,
        "temperature": 0,
        "word_timestamps": True,
        "condition_on_previous_text": False
    }

    print("\n📋 Lightning Whisper MLX kwargs:")
    for key, expected in expected_lightning.items():
        actual = lightning_kwargs.get(key, "MISSING")
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {key}: {actual}")

    # Lightning initialization kwargs
    lightning_init = config.get_lightning_whisper_mlx_init_kwargs()
    expected_init = {
        "model": "base",  # for default base model
        "batch_size": 8,
        "quant": "8bit"
    }

    print("\n📋 Lightning Whisper MLX init kwargs:")
    for key, expected in expected_init.items():
        actual = lightning_init.get(key, "MISSING")
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {key}: {actual}")

    # Test MLX model repository paths
    print("\n📋 MLX Model Repository Paths:")
    mlx_repo = config.get_mlx_model_repo()
    expected_repo = "mlx-community/whisper-base-mlx-q4"
    status = "✅" if mlx_repo == expected_repo else "❌"
    print(f"  {status} base model: {mlx_repo}")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CRITICAL PARAMETERS MATCH PROVIDED EXAMPLES!")
    else:
        print("⚠️  SOME PARAMETERS DON'T MATCH - REVIEW ABOVE")

    return all_passed


def main():
    """Main test function."""
    print_config_summary()
    print("\n")
    test_passed = test_config_values()

    if test_passed:
        print("\n✅ Configuration test PASSED - Ready for apples-to-apples benchmark!")
    else:
        print("\n❌ Configuration test FAILED - Review discrepancies above")

    return test_passed


if __name__ == "__main__":
    main()
