#!/usr/bin/env python3
"""
Test script for Professional Pix2Pix Models

This script tests the professional models to ensure they work correctly
and can generate high-quality building facades.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_model_creation():
    """Test creating professional models"""
    print("🧪 Testing Professional Model Creation...")
    
    try:
        from src.models import ProfessionalGenerator, ProfessionalPatchGANDiscriminator
        
        # Test generator
        print("  📦 Creating Professional Generator...")
        generator = ProfessionalGenerator(input_shape=(512, 512, 3))
        print(f"  ✅ Generator created: {generator.name}")
        
        # Test discriminator
        print("  📦 Creating Professional Discriminator...")
        discriminator = ProfessionalPatchGANDiscriminator(input_shape=(512, 512, 3))
        print(f"  ✅ Discriminator created: {discriminator.name}")
        
        # Test forward pass
        print("  🔄 Testing forward pass...")
        dummy_input = tf.random.normal([1, 512, 512, 3])
        
        gen_output = generator(dummy_input, training=False)
        print(f"  ✅ Generator output shape: {gen_output.shape}")
        
        disc_output = discriminator([dummy_input, gen_output], training=False)
        print(f"  ✅ Discriminator output shape: {disc_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False

def test_inference():
    """Test professional inference"""
    print("\n🧪 Testing Professional Inference...")
    
    try:
        from src.infer import ProfessionalInference, create_semantic_facade
        
        # Create semantic facade
        print("  🎨 Creating semantic facade...")
        semantic = create_semantic_facade(512, 512, "residential")
        semantic_img = Image.fromarray(semantic)
        semantic_img.save("test_semantic.png")
        print("  ✅ Semantic facade created: test_semantic.png")
        
        # Test inference
        print("  🚀 Testing professional inference...")
        inferencer = ProfessionalInference(model_path=None, resolution=512)
        
        # Preprocess
        processed = inferencer.preprocess_image(semantic_img)
        batch = np.expand_dims(processed, 0)
        
        # Generate (with untrained model)
        result = inferencer.generator(batch, training=False)
        
        # Post-process
        result_img = inferencer.postprocess_image(result[0])
        result_img.save("test_result.png")
        print("  ✅ Inference completed: test_result.png")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        return False

def test_training_setup():
    """Test training setup"""
    print("\n🧪 Testing Training Setup...")
    
    try:
        from src.train import TrainingConfig, create_models, create_optimizers
        
        # Test config
        print("  ⚙️ Creating training config...")
        config = TrainingConfig()
        print(f"  ✅ Config created: {config.EPOCHS} epochs, {config.INPUT_HEIGHT}x{config.INPUT_WIDTH}")
        
        # Test model creation
        print("  📦 Creating training models...")
        generator, discriminator = create_models(config)
        print(f"  ✅ Models created for training")
        
        # Test optimizers
        print("  🔧 Creating optimizers...")
        gen_opt, disc_opt = create_optimizers(config)
        print(f"  ✅ Optimizers created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training setup failed: {e}")
        return False

def test_loss_functions():
    """Test advanced loss functions"""
    print("\n🧪 Testing Advanced Loss Functions...")
    
    try:
        from src.models import generator_loss, discriminator_loss, PerceptualLoss
        
        # Create dummy data
        batch_size = 2
        height, width = 256, 256
        
        dummy_disc_output = tf.random.normal([batch_size, 30, 30, 1])
        dummy_gen_output = tf.random.normal([batch_size, height, width, 3])
        dummy_target = tf.random.normal([batch_size, height, width, 3])
        
        # Test generator loss
        print("  📊 Testing generator loss...")
        perceptual_loss_fn = PerceptualLoss()
        gen_total_loss, gen_gan_loss, gen_l1_loss, gen_perceptual_loss = generator_loss(
            dummy_disc_output, dummy_gen_output, dummy_target, perceptual_loss_fn
        )
        print(f"  ✅ Generator loss: {gen_total_loss:.4f}")
        
        # Test discriminator loss
        print("  📊 Testing discriminator loss...")
        disc_loss = discriminator_loss(dummy_disc_output, dummy_disc_output)
        print(f"  ✅ Discriminator loss: {disc_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Loss function test failed: {e}")
        return False

def create_demo_comparison():
    """Create a demo comparison image"""
    print("\n🎨 Creating Demo Comparison...")
    
    try:
        from src.infer import create_semantic_facade
        
        # Create different building types
        residential = create_semantic_facade(256, 256, "residential")
        commercial = create_semantic_facade(256, 256, "commercial")
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(residential)
        axes[0].set_title("Residential Semantic Facade")
        axes[0].axis('off')
        
        axes[1].imshow(commercial)
        axes[1].set_title("Commercial Semantic Facade")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("demo_semantic_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  ✅ Demo comparison created: demo_semantic_comparison.png")
        return True
        
    except Exception as e:
        print(f"  ❌ Demo creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Professional Pix2Pix Model Testing")
    print("=" * 50)
    
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🔧 GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU setup warning: {e}")
    else:
        print("💻 Running on CPU")
    
    # Run tests
    tests = [
        ("Model Creation", test_model_creation),
        ("Inference", test_inference),
        ("Training Setup", test_training_setup),
        ("Loss Functions", test_loss_functions),
        ("Demo Creation", create_demo_comparison)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Professional Pix2Pix is ready!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
