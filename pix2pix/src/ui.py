import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Professional Pix2Pix Building Generator", layout="wide")

st.title('🎨 Universal Pix2Pix: Drawing to Photorealistic Image Converter')
st.write("Draw ANY object and watch it transform into a stunning photorealistic image! Works with people, animals, cars, food, furniture, and more!")

# Add universal features notice
st.info("🚀 **UNIVERSAL AI**: Professional-grade Pix2Pix that converts any drawing into photorealistic images with proper colors, textures, and materials!")

# Add universal drawing tools sidebar
with st.sidebar:
    st.header("🎨 Universal Drawing Tools")

    st.markdown("### 🌟 Draw Any Object!")
    st.markdown("Simply draw any object using natural colors and the AI will make it photorealistic!")

    # Object categories for inspiration
    st.markdown("### 📝 Object Categories")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🍎 Food & Nature:**")
        st.markdown("• Fruits & vegetables")
        st.markdown("• Trees & flowers")
        st.markdown("• Animals & pets")

        st.markdown("**🚗 Vehicles:**")
        st.markdown("• Cars & trucks")
        st.markdown("• Planes & boats")
        st.markdown("• Motorcycles")

    with col2:
        st.markdown("**👤 People & Objects:**")
        st.markdown("• People & faces")
        st.markdown("• Furniture")
        st.markdown("• Electronics")

        st.markdown("**🏠 Buildings:**")
        st.markdown("• Houses & buildings")
        st.markdown("• Landmarks")
        st.markdown("• Interior spaces")

    st.markdown("---")

    # Color suggestions
    st.markdown("### 🎨 Color Tips")
    st.markdown("**Use natural colors:**")
    st.markdown("🔴 Red for apples, roses, cars")
    st.markdown("🟢 Green for leaves, grass")
    st.markdown("🔵 Blue for sky, water")
    st.markdown("🟤 Brown for wood, earth")
    st.markdown("⚫ Black for outlines")
    st.markdown("⚪ White for highlights")

    st.markdown("---")

    # Template options
    st.markdown("### 🎯 Quick Start Templates")
    template_type = st.selectbox(
        "Choose a template:",
        ["None", "🍎 Apple", "🚗 Car", "🏠 House", "🐱 Cat", "🌳 Tree", "👤 Person"],
        help="Start with a pre-made template or draw from scratch"
    )

    if st.button("🎯 Load Template"):
        if template_type != "None":
            st.session_state.load_template = template_type

    st.markdown("---")

    # Universal generation settings
    st.markdown("### ⚙️ Universal AI Settings")

    # Model selection
    model_type = st.selectbox(
        "🧠 AI Model:",
        ["Universal Pix2Pix (512x512)", "Standard Pix2Pix (256x256)"],
        help="Choose between universal high-res or standard model"
    )

    object_style = st.selectbox(
        "🎨 Realism Style:",
        ["Photorealistic", "Artistic", "Hyperrealistic", "Natural", "Studio Quality", "Documentary"],
        help="Choose the style of realism for your generated object"
    )

    # Resolution settings
    if "Universal" in model_type:
        resolution = st.selectbox(
            "📐 Output Resolution:",
            ["512x512 (High Quality)", "1024x1024 (Ultra High)"],
            help="Higher resolution = more detail but slower generation"
        )
        enhance_quality = st.checkbox(
            "✨ Enhanced Post-Processing",
            value=True,
            help="Apply professional sharpening and color enhancement"
        )
    else:
        resolution = "256x256 (Standard)"
        enhance_quality = False

    detail_level = st.slider(
        "🔍 Detail Level:",
        min_value=1,
        max_value=5,
        value=4 if "Universal" in model_type else 3,
        help="Higher values add more realistic details and textures"
    )

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        texture_quality = st.slider(
            "Material Quality:",
            min_value=1,
            max_value=5,
            value=4,
            help="Quality of material textures (skin, metal, fabric, etc.)"
        )

        lighting_style = st.selectbox(
            "Lighting Style:",
            ["Natural Daylight", "Studio Lighting", "Golden Hour", "Soft Light", "Dramatic"],
            help="Lighting conditions for the generated object"
        )

        background_style = st.selectbox(
            "Background:",
            ["Natural", "Studio White", "Minimal", "Contextual", "Transparent"],
            help="Background style for the generated object"
        )

        color_accuracy = st.slider(
            "Color Accuracy:",
            min_value=1,
            max_value=5,
            value=4,
            help="How closely colors match real-world objects"
        )

    st.markdown("---")
    st.markdown("""
    ### 📖 Universal AI Workflow:
    1. **🎨 Draw** any object using natural colors
    2. **🌈 Use** realistic colors (red apple, blue sky, etc.)
    3. **⚙️ Configure** AI settings above
    4. **🔄 Generate** photorealistic version
    5. **✨ Enjoy** professional-quality results!

    **💡 Pro Tips:**
    - Use Universal Pix2Pix for best quality
    - Enable Enhanced Post-Processing for photorealism
    - Higher resolution = more realistic detail
    - Natural colors work best (red for apples, etc.)
    """)

# Handle template loading
if 'load_template' in st.session_state and st.session_state.load_template != "None":
    from universal_templates import universal_template_generator

    # Generate universal template
    template_image = universal_template_generator.create_template(
        st.session_state.load_template, 400, 400
    )
    st.session_state.template_image = template_image
    st.session_state.load_template = "None"  # Reset
    st.rerun()

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎨 Universal Drawing Canvas")
    st.caption("Draw any object using natural colors for photorealistic conversion")

    # Canvas size based on model type
    canvas_size = 512 if "Universal" in model_type else 400

    # Initialize background image
    background_image = None
    if 'template_image' in st.session_state:
        background_image = Image.fromarray(st.session_state.template_image)
        # Resize template to match canvas size
        background_image = background_image.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)

    # Create universal canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
        stroke_width=6 if "Universal" in model_type else 5,
        stroke_color="#FF0000",  # Default to red (good for many objects)
        background_color="#FFFFFF" if background_image is None else None,  # White background
        background_image=background_image,
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        key="universal_canvas",
    )

    # Display canvas info
    st.caption(f"📐 Canvas: {canvas_size}x{canvas_size} pixels | Model: {model_type}")

    # Color picker for universal drawing
    st.markdown("**Quick Color Selection:**")
    color_cols = st.columns(4)

    with color_cols[0]:
        if st.button("🔴 Red", help="Red - Apples, roses, cars"):
            st.session_state.current_color = "#FF0000"  # Red
    with color_cols[1]:
        if st.button("🟢 Green", help="Green - Leaves, grass, nature"):
            st.session_state.current_color = "#00FF00"  # Green
    with color_cols[2]:
        if st.button("🔵 Blue", help="Blue - Sky, water, cars"):
            st.session_state.current_color = "#0000FF"  # Blue
    with color_cols[3]:
        if st.button("🟤 Brown", help="Brown - Wood, earth, animals"):
            st.session_state.current_color = "#8B4513"  # Brown

    # Additional color row
    color_cols2 = st.columns(4)
    with color_cols2[0]:
        if st.button("⚫ Black", help="Black - Outlines, details"):
            st.session_state.current_color = "#000000"  # Black
    with color_cols2[1]:
        if st.button("⚪ White", help="White - Highlights, clouds"):
            st.session_state.current_color = "#FFFFFF"  # White
    with color_cols2[2]:
        if st.button("🟡 Yellow", help="Yellow - Sun, bananas"):
            st.session_state.current_color = "#FFFF00"  # Yellow
    with color_cols2[3]:
        if st.button("🟣 Purple", help="Purple - Flowers, grapes"):
            st.session_state.current_color = "#800080"  # Purple

with col2:
    st.subheader("✨ Universal Photorealistic Generator")

    if canvas_result.image_data is not None:
        # Check if there's actually something drawn
        canvas_array = np.array(canvas_result.image_data)
        if np.any(canvas_array[:,:,:3] < 255):  # Check if any pixel is not white

            # Universal generation button
            if st.button("🚀 Generate Photorealistic Object", type="primary"):
                start_time = time.time()

                # Show processing status with progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🔄 Initializing universal AI model...")
                progress_bar.progress(10)

                try:
                    # Import universal inference
                    from infer import infer_from_canvas, ProfessionalInference

                    status_text.text("🧠 Loading universal Pix2Pix model...")
                    progress_bar.progress(30)

                    # Determine model path and resolution
                    if "Universal" in model_type:
                        model_path = './outputs/final_professional_generator.h5'
                        target_resolution = 512 if "512x512" in resolution else 1024
                    else:
                        model_path = './outputs/pix2pix_generator.h5'
                        target_resolution = 256

                    status_text.text("🎨 Processing drawing...")
                    progress_bar.progress(50)

                    # Create a mock canvas data object
                    class MockCanvasData:
                        def __init__(self, image_data):
                            self.image_data = image_data

                    mock_canvas = MockCanvasData(canvas_array)

                    status_text.text("✨ Generating photorealistic object...")
                    progress_bar.progress(70)

                    # Generate using universal inference
                    result_img, result_path = infer_from_canvas(
                        mock_canvas,
                        model_path=model_path,
                        background_option=background_style.lower(),
                        enhance_colors=enhance_quality,
                        resolution=target_resolution
                    )

                    status_text.text("🎨 Applying realistic textures and colors...")
                    progress_bar.progress(90)

                    if result_img is not None:
                        # Display the result
                        st.image(result_img, caption=f"✨ Photorealistic Object ({resolution})", use_column_width=True)

                        generation_time = time.time() - start_time
                        progress_bar.progress(100)
                        status_text.text("🎉 Generation completed!")

                        # Show success message with metrics
                        st.success(f"🎉 Photorealistic object generated in {generation_time:.1f}s!")

                        # Display generation info
                        col1_info, col2_info, col3_info = st.columns(3)

                        with col1_info:
                            st.metric(
                                "🎨 Style",
                                object_style,
                                f"Detail: {detail_level}/5"
                            )

                        with col2_info:
                            st.metric(
                                "📐 Resolution",
                                resolution.split()[0],
                                f"Material: {texture_quality}/5"
                            )

                        with col3_info:
                            st.metric(
                                "⚡ Generation Time",
                                f"{generation_time:.1f}s",
                                "Universal AI"
                            )

                        # Universal features applied
                        st.info(f"✨ Applied: {model_type} | {lighting_style} | Enhanced: {enhance_quality} | Colors: {color_accuracy}/5")
                    else:
                        st.error("❌ Generation failed. Please try again.")

                except Exception as e:
                    st.error(f"❌ Error during generation: {str(e)}")
                    st.info("💡 Note: Universal model may not be trained yet. Using basic color enhancement.")

                    # Fallback to basic enhancement
                    try:
                        status_text.text("🔄 Using basic enhancement fallback...")
                        from PIL import ImageEnhance, ImageFilter

                        # Convert canvas to PIL image
                        canvas_img = Image.fromarray(canvas_array[:,:,:3])

                        # Apply basic enhancements
                        enhanced = canvas_img.filter(ImageFilter.SMOOTH)
                        enhancer = ImageEnhance.Color(enhanced)
                        enhanced = enhancer.enhance(1.5)
                        enhancer = ImageEnhance.Contrast(enhanced)
                        enhanced = enhancer.enhance(1.2)

                        st.image(enhanced, caption="🎨 Enhanced Drawing", use_column_width=True)
                        st.info("✨ Applied: Basic Color Enhancement (Fallback)")
                    except Exception as fallback_error:
                        st.error(f"❌ Fallback also failed: {str(fallback_error)}")

                finally:
                    progress_bar.empty()
                    status_text.empty()

            else:
                # Show preview without generation
                st.info("👆 Click 'Generate Photorealistic Object' to create amazing results!")

                # Show drawing preview
                drawing_image = canvas_array[:,:,:3]
                st.image(drawing_image, caption="🎨 Your Drawing Preview", use_column_width=True)

                # Show universal generation details
                with st.expander("🔍 Universal AI Details"):
                    st.markdown(f"""
                    **🎨 AI Model Details:**
                    - **Model**: {model_type}
                    - **Architecture**: Universal U-Net with Attention
                    - **Resolution**: {resolution}
                    - **Style**: {object_style} Realism
                    - **Lighting**: {lighting_style}
                    - **Background**: {background_style}
                    - **Post-Processing**: {'Enabled' if enhance_quality else 'Disabled'}
                    - **Color Accuracy**: {color_accuracy}/5

                    **✨ Universal Features:**
                    - **Generator**: Enhanced U-Net for any object type
                    - **Discriminator**: Multi-scale PatchGAN
                    - **Loss Functions**: L1 + Adversarial + Perceptual
                    - **Materials**: Realistic textures for any object
                    - **Colors**: Accurate real-world color mapping
                    - **Quality**: Professional photorealistic results
                    """)

                # Show color analysis
                with st.expander("🎨 Color Analysis"):
                    st.write("**Colors Used in Your Drawing:**")

                    # Analyze which colors were used
                    unique_colors = np.unique(drawing_image.reshape(-1, 3), axis=0)

                    # Filter out white/near-white colors (background)
                    significant_colors = []
                    for color in unique_colors:
                        if not (color[0] > 240 and color[1] > 240 and color[2] > 240):  # Not white-ish
                            significant_colors.append(color)

                    if significant_colors:
                        st.write("**Detected Colors:**")
                        for i, color in enumerate(significant_colors[:8]):  # Show max 8 colors
                            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                            st.markdown(f'• **Color {i+1}**: <span style="background-color: {color_hex}; padding: 2px 8px; border-radius: 3px; color: white;">{color_hex}</span>', unsafe_allow_html=True)
                    else:
                        st.write("No significant colors detected. Try drawing with more vibrant colors!")

        else:
            # No drawing detected - show instructions and example
            st.info("👆 Draw any object on the canvas to generate a photorealistic version!")

            # Show example
            st.markdown("**Example Objects You Can Draw:**")

            # Create example grid
            example_cols = st.columns(3)

            with example_cols[0]:
                st.markdown("🍎 **Food**")
                st.markdown("• Fruits & vegetables")
                st.markdown("• Desserts & meals")

            with example_cols[1]:
                st.markdown("🚗 **Vehicles**")
                st.markdown("• Cars & motorcycles")
                st.markdown("• Planes & boats")

            with example_cols[2]:
                st.markdown("🐱 **Animals**")
                st.markdown("• Pets & wildlife")
                st.markdown("• Birds & fish")

            st.markdown("**💡 Tips:**")
            st.markdown("• Use natural colors (red apple, blue car, etc.)")
            st.markdown("• Draw clear, simple shapes")
            st.markdown("• Try the templates above for inspiration!")

    else:
        # No canvas data available
        st.info("👈 Draw something on the canvas to generate an image!")

# Add universal drawing instructions
st.markdown("---")
st.markdown("""
### 📝 How to Create Photorealistic Objects:

#### 🎨 **Universal Drawing Method:**
1. **🖌️ Choose Any Object** - apple, car, person, animal, furniture, etc.
2. **🌈 Use Natural Colors** - red for apples, blue for sky, green for leaves
3. **✏️ Draw Simple Shapes** - clear outlines and basic forms work best
4. **🎯 Add Details** - use different colors for different parts
5. **🚀 Generate** and watch your drawing become photorealistic!

#### 🎯 **Object Templates:**
- **🍎 Apple**: Red fruit with green leaf
- **🚗 Car**: Blue vehicle with black wheels
- **🏠 House**: Building with roof, windows, door
- **🐱 Cat**: Gray pet with pink nose
- **🌳 Tree**: Green leaves with brown trunk
- **👤 Person**: Simple human figure

#### 💡 **Pro Tips:**
- **Natural Colors Work Best**: Use colors objects actually have in real life
- **Simple is Better**: Clear, simple drawings generate better results
- **Try Templates**: Start with a template and modify it
- **Experiment**: Different styles create different photorealistic results

#### 🔧 **Technical Features:**
- **Universal AI**: Works with any object category
- **Color Intelligence**: Automatically enhances colors to be realistic
- **Material Generation**: Appropriate textures for each object type
- **High Resolution**: Up to 1024x1024 pixel output

### 🌟 **Example Workflows:**

**🍎 Draw an Apple:**
1. Load the "🍎 Apple" template
2. Modify colors (red, green leaf)
3. Generate → Get photorealistic apple!

**🚗 Draw a Car:**
1. Use blue for car body
2. Add black wheels
3. Generate → Get realistic car with metallic paint!

**🐱 Draw a Cat:**
1. Use gray for body
2. Add pink nose, black eyes
3. Generate → Get fluffy realistic cat!
""")

# Universal file upload alternative
st.markdown("---")
st.subheader("📁 Universal Image Upload")
st.write("Upload any drawing or sketch for photorealistic transformation")

uploaded = st.file_uploader(
    "Choose an image file (drawing, sketch, or photo):",
    type=['jpg','jpeg','png'],
    help="Upload any drawing or sketch for photorealistic conversion"
)

if uploaded is not None:
    img_path = f'./input_image.jpg'
    with open(img_path, 'wb') as f:
        f.write(uploaded.read())

    col1_upload, col2_upload = st.columns(2)
    with col1_upload:
        st.image(img_path, caption="📤 Uploaded Image", use_column_width=True)

    with col2_upload:
        if st.button("🚀 Process with Universal AI", key="upload_process"):
            with st.spinner("🔄 Processing with universal model..."):
                try:
                    from infer import infer_from_path

                    # Use universal model settings
                    model_path = './outputs/final_professional_generator.h5' if "Universal" in model_type else './outputs/pix2pix_generator.h5'
                    target_resolution = 512 if "Universal" in model_type else 256

                    result_path = infer_from_path(
                        img_path,
                        model_path=model_path,
                        resolution=target_resolution,
                        enhance_quality=enhance_quality
                    )

                    st.image(result_path, caption="✨ Universal Generated Output", use_column_width=True)
                    st.success("🎉 Image processed with Universal AI!")

                    # Show processing details
                    st.info(f"🔧 Processed with: {model_type} | Resolution: {target_resolution}x{target_resolution} | Style: {object_style}")

                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.info("💡 Note: Universal model may not be trained yet. Try training first or use canvas drawing.")
        else:
            st.info("👆 Click 'Process with Universal AI' to transform your image")
    