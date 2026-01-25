

import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import zipfile
import io

from src.online_query import PhotoRetriever
from src.person_manager import PersonManager


# Page configuration
st.set_page_config(
    page_title="PhotoGPT",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .photo-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def draw_bounding_boxes(image_path: str, faces: list) -> np.ndarray:
    """Draw bounding boxes on image for detected faces."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        similarity = face['similarity']
        
        # Draw rectangle
        color = (0, 255, 0)
        thickness = 3
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thickness)
        
        # Add similarity score
        label = f"{similarity:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        cv2.rectangle(img_rgb, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(img_rgb, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return img_rgb


def create_zip(image_paths: list) -> bytes:
    """Create ZIP file from list of image paths."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_path in image_paths:
            if os.path.exists(img_path):
                # Add file to ZIP with just filename
                zip_file.write(img_path, arcname=os.path.basename(img_path))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if 'matches' not in st.session_state:
        st.session_state.matches = None
    if 'registration_success' not in st.session_state:
        st.session_state.registration_success = False
    
    # Modern Header
    st.markdown("""
    <div class="main-header">
        <h1>📸 PhotoGPT </h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;"> Find your photos using natural language</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick workflow guide
    
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        with st.expander("🗂️ File Paths", expanded=False):
            index_path = st.text_input("FAISS Index", "data/embeddings/faiss.index")
            metadata_path = st.text_input("Metadata", "data/embeddings/metadata.json")
            profiles_path = st.text_input("Profiles", "data/embeddings/person_profiles.json")
        
        st.markdown("### 🎚️ Match Sensitivity")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.3,
            max_value=0.8,
            value=0.5,
            step=0.05,
            help="Higher = stricter matching (0.5 recommended)"
        )
        
        # Visual indicator for threshold
        if similarity_threshold >= 0.6:
            st.info("🎯 High precision (fewer matches)")
        elif similarity_threshold >= 0.45:
            st.success("✅ Balanced (recommended)")
        else:
            st.warning("🔍 More results (lower precision)")
        
        st.divider()
        
        # Navigation with enhanced styling
        st.markdown("### 📑 Navigation")
        tab_selection = st.radio(
            "Choose action:",
            ["🔍 Search by Name", "🎨 Semantic Search", "👤 Register New Person"],
            label_visibility="collapsed"
        )
    
    # Check if index exists
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        st.error("❌ Index files not found! Please run offline indexing first.")
        st.code("python src/offline_indexing.py --event-photos-dir data/event_photos --output-dir data/embeddings")
        return
    
    # Initialize managers
    try:
        person_manager = PersonManager(profiles_path)
    except Exception as e:
        st.error(f"Error loading person manager: {str(e)}")
        return
    
    # TAB: Register New Person
    if tab_selection == "👤 Register New Person":
        st.markdown("### 👤 Register New Person")
        st.markdown("🎨 Upload a clear selfie and enter your name to join the database.")
        
        # Show registered people with enhanced UI
        registered_names = person_manager.get_all_names()
        if registered_names:
            st.markdown(f"**✅ Currently Registered ({len(registered_names)}):**")
            # Create nice pills for names
            name_pills = " · ".join([f"**{name}**" for name in registered_names])
            st.markdown(f"<div style='background-color: #e7f3ff; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>{name_pills}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Registration form with enhanced layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 1️⃣ Upload Your Selfie")
            st.caption("📷 Choose a clear, front-facing photo")
            uploaded_file = st.file_uploader(
                "Choose image",
                type=['jpg', 'jpeg', 'png'],
                key="registration_selfie",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="✅ Preview", use_container_width=True)
            else:
                st.info("Click above to upload your selfie")
        
        with col2:
            st.markdown("#### 2️⃣ Enter Your Details")
            st.caption("This name will be used for searching")
            person_name = st.text_input(
                "Name",
                placeholder="e.g., John Smith",
                help="Enter your full name as you want it to appear",
                key="person_name_input"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if uploaded_file is not None and person_name:
                if st.button("✅ Register Me", type="primary", use_container_width=True):
                    # Save selfie
                    selfies_dir = "data/selfies"
                    os.makedirs(selfies_dir, exist_ok=True)
                    
                    # Create unique filename
                    safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in person_name)
                    selfie_filename = f"{safe_name}_{uploaded_file.name}"
                    selfie_path = os.path.join(selfies_dir, selfie_filename)
                    
                    with open(selfie_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Register person
                    with st.spinner(f"Registering {person_name}..."):
                        result = person_manager.register_person(person_name, selfie_path)
                    
                    if result['success']:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>🎉 Registration Successful!</h3>
                            <p>{result['message']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                        st.session_state.registration_success = True
                    else:
                        st.error(f"❌ {result['message']}")
            elif uploaded_file is not None:
                st.warning("Please enter your name")
            elif person_name:
                st.warning("Please upload a selfie")
    
    # TAB: Semantic Search
    elif tab_selection == "🎨 Semantic Search":
        st.markdown("### 🎨 Semantic Photo Search")
        st.markdown("🔍 Describe what you're looking for in natural language - no face registration needed!")
        
        # Semantic search input
        st.markdown("#### ✍️ Enter Your Query")
        col1, col2, col3 = st.columns([5, 2, 1])
        
        with col1:
            semantic_query = st.text_input(
                "Describe what you're looking for",
                value=st.session_state.get('semantic_query', ''),
                placeholder="e.g., people dancing at sunset, group photo outdoors, eating food...",
                key="semantic_query_input",
                label_visibility="collapsed"
            )
        
        with col2:
            semantic_search_button = st.button("🔍 Search", type="primary", use_container_width=True, key="semantic_search_btn")
        
        with col3:
            if st.button("🔄", use_container_width=True, help="Clear search", key="clear_semantic"):
                st.session_state.semantic_matches = None
                st.session_state.semantic_query = ""
                st.rerun()
        
    
        
        # Perform semantic search
        if semantic_search_button and semantic_query:
            st.session_state.semantic_query = semantic_query
            
            # Initialize retriever
            try:
                with st.spinner("Loading AI vision models..."):
                    retriever = PhotoRetriever(index_path, metadata_path)
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                return
            
            # Check if index was built in full_image mode
            if len(retriever.index_manager.metadata) > 0:
                mode = retriever.index_manager.metadata[0].get('mode', 'face')
                if mode != 'full_image':
                    st.error("❌ Semantic search requires the index to be built in 'full_image' mode!")
                    st.info("💡 To enable semantic search, rebuild your index:")
                    st.code("python src/offline_indexing.py --event-photos-dir data/event_photos --output-dir data/embeddings --mode full_image")
                    return
            
            # Perform semantic search
            with st.spinner(f"Searching for: '{semantic_query}'..."):
                st.info(f"🔍 Analyzing {retriever.index_manager.index.ntotal} photos with AI vision...")
                
                result = retriever.find_photos(
                    text_query=semantic_query,
                    similarity_threshold=similarity_threshold
                )
                st.session_state.semantic_matches = result
            st.rerun()
        
        # Display semantic search results
        if st.session_state.get('semantic_matches') is not None:
            result = st.session_state.semantic_matches
            
            st.divider()
            st.subheader("📸 Matching Photos")
            
            if not result['success']:
                st.warning(result['message'])
                st.info("💡 Tips to improve results:")
                st.markdown("""
                - Try lowering the similarity threshold in the sidebar
                - Use different descriptive words
                - Try broader or more specific queries
                - Ensure your index was built in 'full_image' mode
                """)
            else:
                # Success message
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ {result['message']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                query_text = result['query_info'].get('query_text', 'Unknown')
                
                # Enhanced Stats Cards
                st.markdown("#### 📊 Search Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                        <h2 style='margin: 0;'>{}</h2>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Photos Found</p>
                    </div>
                    """.format(result['total_photos']), unsafe_allow_html=True)
                with col2:
                    avg_sim = sum(p['similarity'] for p in result['matches']) / len(result['matches'])
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                        <h2 style='margin: 0;'>{:.1%}</h2>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Avg Relevance</p>
                    </div>
                    """.format(avg_sim), unsafe_allow_html=True)
                with col3:
                    best_sim = max(p['similarity'] for p in result['matches'])
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                        <h2 style='margin: 0;'>{:.1%}</h2>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Best Match</p>
                    </div>
                    """.format(best_sim), unsafe_allow_html=True)
                
                st.caption(f"🔍 Query: **\"{query_text}\"**")
                
                # Download all button
                if len(result['matches']) > 0:
                    all_paths = [p['image_path'] for p in result['matches']]
                    zip_data = create_zip(all_paths)
                    
                    query_safe = semantic_query.replace(' ', '_')[:30]
                    
                    st.download_button(
                        label=f"⬇️ Download All {result['total_photos']} Matching Photos as ZIP",
                        data=zip_data,
                        file_name=f"semantic_search_{query_safe}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
                
                st.divider()
                st.markdown(f"### 📸 Photo Gallery ({result['total_photos']} results)")
                
                # Display photos
                for i, photo in enumerate(result['matches']):
                    with st.container():
                        st.markdown("<div class='photo-card'>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([2.5, 1])
                        
                        with col1:
                            try:
                                if os.path.exists(photo['image_path']):
                                    img = Image.open(photo['image_path'])
                                    st.image(img, caption=f"Photo {i+1} - {os.path.basename(photo['image_path'])}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                        
                        with col2:
                            st.markdown(f"##### 🖼️ Photo #{i+1}")
                            st.markdown(f"**🎯 Relevance:** {photo['similarity']:.1%}")
                            
                            # Progress bar for relevance
                            st.progress(photo['similarity'])
                            
                            st.markdown(f"**📄 File:** {os.path.basename(photo['image_path'])}")
                            
                            # Download button
                            if os.path.exists(photo['image_path']):
                                with open(photo['image_path'], 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Download",
                                        data=f.read(),
                                        file_name=os.path.basename(photo['image_path']),
                                        mime="image/jpeg",
                                        key=f"dl_semantic_{i}",
                                        use_container_width=True
                                    )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB: Search by Name
    elif tab_selection == "🔍 Search by Name":
        st.markdown("### 🔍 Find Event Photos by Name")
        st.markdown("a name below or type to search for someone's event photos.")
        
        # Get registered names
        registered_names = person_manager.get_all_names()
        
        if not registered_names:
            st.warning("⚠️ No registered people yet! Switch to 'Register New Person' to get started.")
            st.info("💡 Once people are registered, you can instantly search for all their event photos.")
            return
        
        # Enhanced quick search buttons
        st.markdown("#### ⚡ Quick Search")
        st.caption(f"Select from {len(registered_names)} registered person(s)")
        
        # Create responsive grid
        num_cols = min(3, len(registered_names))
        cols = st.columns(num_cols)
        for idx, name in enumerate(registered_names):
            with cols[idx % num_cols]:
                if st.button(f"👤 {name}", key=f"quick_{name}", use_container_width=True, type="secondary"):
                    st.session_state.search_name = name
                    st.rerun()
        
        st.divider()
        
        # Search input
        st.markdown("#### 🔎 Search")
        
        col1, col2, col3 = st.columns([5, 2, 1])
        
        with col1:
            search_name = st.text_input(
                "Enter name",
                value=st.session_state.get('search_name', ''),
                placeholder="Type a person's name...",
                key="search_name_input",
                label_visibility="collapsed"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        with col3:
            if st.button("🔄", use_container_width=True, help="Clear search"):
                st.session_state.matches = None
                st.session_state.search_name = ""
                st.session_state.activity_query = ""
                st.rerun()
        
        # Perform search
        if search_button and search_name:
            st.session_state.search_name = search_name
            
            # Get person's embedding from registered profile
            profile = person_manager.get_profile(search_name)
            
            if profile is None:
                st.error(f"❌ '{search_name}' is not registered. Please check the name or register first.")
                st.info("💡 Tip: Go to 'Register New Person' tab to register this person first.")
            else:
                st.info(f"✓ Found registration for **{profile['name']}**")
                
                # Show registered selfie
                with st.expander("View Registered Selfie"):
                    if os.path.exists(profile['selfie_path']):
                        reg_img = Image.open(profile['selfie_path'])
                        st.image(reg_img, caption=f"Registered selfie of {profile['name']}", width=200)
                
                # Initialize retriever
                try:
                    with st.spinner("Loading face recognition models..."):
                        retriever = PhotoRetriever(index_path, metadata_path)
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
                    return
                
                # Regular face search - all photos
                with st.spinner(f"Searching event photos for {profile['name']}..."):
                    st.info(f"🔍 Matching {profile['name']}'s face against {retriever.index_manager.index.ntotal} indexed faces from event photos...")
                    
                    result = retriever.find_photos_by_embedding(
                        query_embedding=profile['embedding'],
                        similarity_threshold=similarity_threshold,
                        person_name=profile['name']
                    )
                    st.session_state.matches = result
                st.rerun()
        
        # Display results
        if st.session_state.matches is not None:
            result = st.session_state.matches
            
            st.divider()
            st.subheader("📸 Event Photos Containing This Person")
            
            if not result['success']:
                st.warning(result['message'])
                st.info("💡 Tips to improve results:")
                st.markdown("""
                - Try lowering the similarity threshold in the sidebar
                - Make sure the registered selfie is clear and well-lit
                - Verify the person actually appears in the event photos
                """)
            else:
                # Success message with details
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ {result['message']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                person_searched = result['query_info'].get('person_name', 'Unknown')
                
                # Enhanced Stats Cards
                st.markdown("#### 📊 Search Results Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                        <h2 style='margin: 0;'>{}</h2>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Photos Found</p>
                    </div>
                    """.format(result['total_photos']), unsafe_allow_html=True)
                with col2:
                    avg_sim = sum(p['max_similarity'] for p in result['matches']) / len(result['matches'])
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                        <h2 style='margin: 0;'>{:.1%}</h2>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Avg Confidence</p>
                    </div>
                    """.format(avg_sim), unsafe_allow_html=True)
                with col3:
                    best_sim = max(p['max_similarity'] for p in result['matches'])
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                        <h2 style='margin: 0;'>{:.1%}</h2>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Best Match</p>
                    </div>
                    """.format(best_sim), unsafe_allow_html=True)
                
                # Download all button
                if len(result['matches']) > 0:
                    all_paths = [p['image_path'] for p in result['matches']]
                    zip_data = create_zip(all_paths)
                    
                    person_name_safe = st.session_state.get('search_name', 'photos').replace(' ', '_')
                    
                    st.download_button(
                        label=f"⬇️ Download All {result['total_photos']} Photos of {person_searched} as ZIP",
                        data=zip_data,
                        file_name=f"{person_name_safe}_event_photos.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
                
                st.divider()
                
                # Check if this is activity search
                is_activity_search = result['query_info'].get('query_type') == 'person_activity'
                activity_name = result['query_info'].get('activity', '') if is_activity_search else None
                
                if is_activity_search and activity_name:
                    st.markdown(f"### 📸 Photos of {person_searched} {activity_name} ({result['total_photos']} results)")
                else:
                    st.markdown(f"### 📸 Event Photos Gallery ({result['total_photos']} results)")
                
                # Display photos with enhanced cards
                for i, photo in enumerate(result['matches']):
                    with st.container():
                        st.markdown("<div class='photo-card'>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([2.5, 1])
                        
                        with col1:
                            try:
                                # For activity search, show full image; for face search, show bounding boxes
                                if is_activity_search or 'faces' not in photo:
                                    if os.path.exists(photo['image_path']):
                                        img = Image.open(photo['image_path'])
                                        st.image(img, caption=f"Photo {i+1} - {os.path.basename(photo['image_path'])}", use_container_width=True)
                                else:
                                    img_with_boxes = draw_bounding_boxes(photo['image_path'], photo['faces'])
                                    st.image(img_with_boxes, caption=f"Event Photo {i+1} - {os.path.basename(photo['image_path'])}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                        
                        with col2:
                            st.markdown(f"##### 🖼️ Photo #{i+1}")
                            st.markdown(f"**🎯 Overall Score:** {photo['max_similarity']:.1%}")
                            
                            # Progress bar for confidence
                            st.progress(photo['max_similarity'])
                            
                            # Show detailed scores for activity search
                            if is_activity_search:
                                if 'face_similarity' in photo and 'activity_similarity' in photo:
                                    st.caption(f"👤 Face match: {photo['face_similarity']:.1%}")
                                    st.caption(f"🎬 Activity match: {photo['activity_similarity']:.1%}")
                            else:
                                st.markdown(f"**👥 Faces Detected:** {photo.get('num_matches', 1)}")
                                
                                # Show face details
                                if photo.get('num_matches', 1) > 1:
                                    st.info(f"💡 Multiple matches in this photo")
                            
                            st.markdown(f"**📄 File:** {os.path.basename(photo['image_path'])}")
                            
                            if 'avg_similarity' in photo and not is_activity_search:
                                st.caption(f"Avg: {photo['avg_similarity']:.1%}")
                            
                            # Download button
                            if os.path.exists(photo['image_path']):
                                with open(photo['image_path'], 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Download",
                                        data=f.read(),
                                        file_name=os.path.basename(photo['image_path']),
                                        mime="image/jpeg",
                                        key=f"dl_{i}",
                                        use_container_width=True
                                    )
                        
                        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()