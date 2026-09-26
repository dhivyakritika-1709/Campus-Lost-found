"""
app.py
Campus Lost & Found Management System
A complete AI/ML-powered platform built for B.Tech CSE students.
Features:
- Dual-direction matching (LOST <-> FOUND) powered by Scikit-Learn Logistic Regression
- Genuine feature comparison with predict_proba()
- Privacy-preserving Contact Request System (Phone numbers never publicly shown)
- SQLite database persistence with image upload support
- Clean, modern, responsive Campus Project UI
"""

import os
import uuid
from datetime import datetime
from PIL import Image
import streamlit as st

import database as db
import matcher
import features

# ---------------------------------------------------------
# Page Configuration & Visual Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Campus Lost & Found | AI Reconnect",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Injected CSS for a polished campus portal aesthetic
st.markdown("""
<style>
    /* Main container and font styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Banner / Hero Card */
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
        max-width: 700px;
    }
    
    /* Metric & Stat Cards */
    .stat-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }
    .stat-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Report & Match Cards */
    .report-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .report-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    }
    
    .badge-lost {
        display: inline-block;
        background: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    .badge-found {
        display: inline-block;
        background: #dcfce7;
        color: #15803d;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    .badge-match {
        display: inline-block;
        background: #e0e7ff;
        color: #4338ca;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
    }
    
    /* Match Card Special Highlight */
    .match-container {
        border-left: 4px solid #4f46e5;
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .match-prob-high {
        font-size: 1.5rem;
        font-weight: 800;
        color: #4338ca;
    }
    
    /* Privacy notice box */
    .privacy-notice {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #166534;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }
    
    .warning-notice {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #92400e;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Ensure database and folders exist
db.init_db()
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def save_uploaded_image(uploaded_file):
    """Saves uploaded image with unique UUID filename to uploads/."""
    if uploaded_file is None:
        return None
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpg"
        unique_name = f"{uuid.uuid4().hex[:12]}{ext}"
        save_path = os.path.join(UPLOADS_DIR, unique_name)
        
        image = Image.open(uploaded_file)
        # Convert RGBA to RGB for jpeg compatibility if needed
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(save_path, optimize=True, quality=85)
        return save_path
    except Exception as e:
        st.warning(f"Note: Could not process uploaded image ({e}). Report will be saved without image.")
        return None


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Campus Lost & Found")
    st.markdown("---")

    navigation_options = [
        "🏠 Home",
        "🔴 Report Lost Item",
        "🟢 Report Found Item",
        "🔎 Find Matches",
        "📋 My Reports & Requests",
        "🌐 All Public Reports",
    ]

    # Keep the selected page separately from the radio widget state.
    # This allows Home quick-action buttons and form submissions to survive
    # Streamlit reruns without triggering StreamlitWidgetAlreadyInstantiatedError.
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "🏠 Home"

    current_page = st.session_state["current_page"]
    navigation_index = navigation_options.index(current_page)

    menu_option = st.radio(
        "Navigation",
        navigation_options,
        index=navigation_index
    )

    # Store manual sidebar changes for the next rerun. This key is NOT the
    # widget key, so Streamlit permits the assignment safely.
    st.session_state["current_page"] = menu_option

    st.markdown("---")
    
    stats = db.get_stats()
    st.markdown(f"""
    **Live Campus Activity**
    - 🔴 Lost Items: **{stats['total_lost']}**
    - 🟢 Found Items: **{stats['total_found']}**
    - 🤝 Contact Requests: **{stats['pending_requests']} pending**
    """)
    
    st.caption("🔒 Contact Privacy: Phone numbers are strictly private and shared only after mutually accepted contact requests.")


# =========================================================
# PAGE 1: 🏠 HOME
# =========================================================
if menu_option == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">Campus Lost & Found</div>
        <div class="hero-subtitle">
            Lost something? Found something? One search One match."
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Action Buttons
    col_act1, col_act2, col_act3 = st.columns(3)
    with col_act1:
        if st.button("🔴 Report a Lost Item", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "🔴 Report Lost Item"
            st.rerun()
    with col_act2:
        if st.button("🟢 Report a Found Item", use_container_width=True):
            st.session_state["current_page"] = "🟢 Report Found Item"
            st.rerun()
    with col_act3:
        if st.button("🔎 Search & Match Items", use_container_width=True):
            st.session_state["current_page"] = "🔎 Find Matches"
            st.rerun()

    # Metric Cards
    st.markdown("### 📊 Platform Statistics")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Lost Reports", f"{stats['total_lost']} items", delta="Active")
    with m_col2:
        st.metric("Total Found Reports", f"{stats['total_found']} items", delta="In Safe Custody")
    with m_col3:
        st.metric("Successful Connections", f"{stats['accepted_requests']} items", delta="Handed Over")

    st.markdown("---")

    st.markdown("### 📌 Recent Public Campus Reports")
    recent = db.get_all_reports()[:4]
    if not recent:
        st.info("No reports submitted yet. Be the first to report a lost or found item!")
    else:
        grid = st.columns(2)
        for idx, r in enumerate(recent):
            with grid[idx % 2]:
                badge_class = "badge-lost" if r["report_type"] == "LOST" else "badge-found"
                st.markdown(f"""
                <div class="report-card">
                    <span class="{badge_class}">{r['report_type']}</span> &nbsp;
                    <span style="color:#64748b; font-size:0.85rem;">{r['category']} • {r['date']}</span>
                    <h4 style="margin: 0.5rem 0 0.3rem 0; color:#0f172a;">{r['item_name']}</h4>
                    <p style="color:#475569; font-size:0.9rem; margin-bottom: 0.5rem;">{r['description']}</p>
                    <p style="font-size:0.85rem; color:#64748b; margin:0;">
                        📍 <b>Location:</b> {r['location']} &nbsp;|&nbsp; 🎨 <b>Colour:</b> {r['colour']}
                    </p>
                    <p style="font-size:0.8rem; color:#059669; margin-top:0.4rem;">
                        🔒 <i>Contact details are private and shared only after a contact request is accepted.</i>
                    </p>
                </div>
                """, unsafe_allow_html=True)


# =========================================================
# PAGE 2: 🔴 REPORT LOST ITEM
# =========================================================
elif menu_option == "🔴 Report Lost Item":
    st.markdown("## 🔴 Report a Lost Item")
    st.markdown("Provide details about the item you lost on campus.")

    st.markdown("""
    <div class="privacy-notice">
        🔒 <b>Student Privacy Guarantee:</b> Your contact number and confidential identifying details are stored privately and NEVER shown in public listings. Finders can only connect with you through our secure Contact Request system.
    </div>
    """, unsafe_allow_html=True)

    with st.form("lost_item_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            item_name = st.text_input("Item Name *", placeholder="e.g., Black Samsung Galaxy Phone, Blue Milton Bottle")
            category = st.selectbox(
                "Category *",
                ["Electronics", "Personal Belongings", "Daily Essentials", "Campus Specific", "Study Materials", "Bags & Luggage"]
            )
            colour = st.text_input("Primary Colour *", placeholder="e.g., Black, Blue, Silver, Navy")
            location = st.text_input("Location Lost *", placeholder="e.g., Central Library 2nd Floor, Canteen, Lab 1")
            date_lost = st.date_input("Date Lost *", value=datetime.now())

        with c2:
            reporter_name = st.text_input("Your Full Name *", placeholder="e.g., Aarav Sharma")
            contact_number = st.text_input("Your Phone Number (Private) *", placeholder="e.g., +91 98765 43210")
            description = st.text_area("General Description *", placeholder="Describe the item's appearance, brand, size, wear and tear...")
            identifying_details = st.text_input(
                "Secret Identifying Detail (Verification Only) *",
                placeholder="e.g., Lock screen wallpaper, roll number on card, dent on bottom rim"
            )
            uploaded_image = st.file_uploader("Upload Image (Optional)", type=["jpg", "jpeg", "png", "webp"])

        st.caption("Please double check your details before submitting.")
        submitted = st.form_submit_button("Submit Lost Report & Run AI Matcher", type="primary", use_container_width=True)

    if submitted:
        if not item_name or not reporter_name or not contact_number:
            st.error("Please fill in all required fields (Item Name, Your Name, and Phone Number).")
        else:
            image_path = save_uploaded_image(uploaded_image)
            report_id = db.create_report(
                report_type="LOST",
                item_name=item_name,
                category=category,
                description=description,
                colour=colour,
                location=location,
                date=str(date_lost),
                identifying_details=identifying_details,
                reporter_name=reporter_name,
                contact_number=contact_number,
                image_path=image_path
            )

            st.success(f"🎉 Lost Report **#{report_id}** created successfully!")
            st.info(f"💡 Save your **Report ID: #{report_id}** to check status and receive contact requests in the **My Reports** section.")

            # Immediate AI/ML Matching against existing FOUND items
            st.markdown("---")
            st.markdown("### 🤖 Finding the closest match...")
            new_report = db.get_report_by_id(report_id, include_private=False)
            found_candidates = db.get_reports_for_matching("FOUND")
            matches = matcher.find_matches_for_report(new_report, found_candidates, threshold=0.25)

            if not matches:
                st.warning("No potential matches found in the current Found items database. Don't worry! Whenever someone reports a matching found item in the future, the system will identify your lost report.")
            else:
                st.markdown(f"**Found {len(matches)} potential matching item(s) in custody:**")
                for m in matches:
                    cand = m["candidate"]
                    st.markdown(f"""
                    <div class="match-container">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-found">FOUND ITEM #{cand['id']}</span> &nbsp;
                                <span class="badge-match">{m['confidence_tier']}</span>
                                <h3 style="margin: 0.4rem 0 0.2rem 0; color:#1e1b4b;">{cand['item_name']}</h3>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size:0.8rem; color:#64748b; font-weight:600;">AI/ML PROBABILITY</div>
                                <div class="match-prob-high">{m['percentage']}%</div>
                            </div>
                        </div>
                        <p style="color:#334155; margin: 0.5rem 0;">{cand['description']}</p>
                        <p style="font-size:0.85rem; color:#475569; margin: 0;">
                            📍 <b>Location Found:</b> {cand['location']} &nbsp;|&nbsp; 🎨 <b>Colour:</b> {cand['colour']} &nbsp;|&nbsp; 📅 <b>Date:</b> {cand['date']}
                        </p>
                        <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;" />
                        <div style="font-size:0.85rem; color:#1e293b;">
                            <b>Why this may match (Grounded ML Feature Evidence):</b>
                            <ul style="margin: 0.3rem 0; padding-left: 1.2rem;">
                                {''.join([f'<li>✓ {reason}</li>' for reason in m['explanations']])}
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Contact Request Trigger directly from match card
                    col_btn1, col_btn2 = st.columns([1, 2])
                    with col_btn1:
                        msg_key = f"req_msg_lost_{report_id}_{cand['id']}"
                        custom_msg = st.text_input("Optional message for finder:", value="Hi, I believe this found item is mine!", key=msg_key)
                        if st.button(f"🤝 Request Contact with Finder (Item #{cand['id']})", key=f"btn_lost_{cand['id']}", type="primary"):
                            ok, resp_msg = db.create_contact_request(report_id, cand['id'], custom_msg)
                            if ok:
                                st.success(resp_msg)
                            else:
                                st.warning(resp_msg)


# =========================================================
# PAGE 3: 🟢 REPORT FOUND ITEM
# =========================================================
elif menu_option == "🟢 Report Found Item":
    st.markdown("## 🟢 Report a Found Item")
    st.markdown("Did you find an item on campus? Help return it to its rightful owner.")

    st.markdown("""
    <div class="privacy-notice">
        🔒 <b>Finder Privacy Protection:</b> Your phone number is kept strictly confidential. The person claiming the item can only contact you if you approve their request in <b>My Reports</b>.
    </div>
    """, unsafe_allow_html=True)

    with st.form("found_item_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            item_name = st.text_input("Item Name *", placeholder="e.g., Stainless Steel Water Flask, College ID Card")
            category = st.selectbox(
                "Category *",
                ["Electronics", "Personal Belongings", "Daily Essentials", "Campus Specific", "Study Materials", "Bags & Luggage"]
            )
            colour = st.text_input("Primary Colour *", placeholder="e.g., Black, Blue, Silver, White")
            location = st.text_input("Location Found *", placeholder="e.g., Central Library Reading Room, Canteen Counter")
            date_found = st.date_input("Date Found *", value=datetime.now())

        with c2:
            finder_name = st.text_input("Your Full Name (Finder / Custodian) *", placeholder="e.g., Rohan Gupta or Security Officer")
            contact_number = st.text_input("Your Phone Number (Private) *", placeholder="e.g., +91 91234 56789")
            description = st.text_area("Item Description *", placeholder="Describe the item found, where it was left, visible features...")
            identifying_details = st.text_input(
                "Distinctive Markings (Optional)",
                placeholder="e.g., Keychain type, serial sticker, specific sticker..."
            )
            uploaded_image = st.file_uploader("Upload Item Photo (Optional)", type=["jpg", "jpeg", "png", "webp"])

        st.caption("Please keep the found item safe or hand it to campus security.")
        submitted = st.form_submit_button("Submit Found Report & Search Lost Reports", type="primary", use_container_width=True)

    if submitted:
        if not item_name or not finder_name or not contact_number:
            st.error("Please fill in all required fields (Item Name, Your Name, and Phone Number).")
        else:
            image_path = save_uploaded_image(uploaded_image)
            report_id = db.create_report(
                report_type="FOUND",
                item_name=item_name,
                category=category,
                description=description,
                colour=colour,
                location=location,
                date=str(date_found),
                identifying_details=identifying_details,
                reporter_name=finder_name,
                contact_number=contact_number,
                image_path=image_path
            )

            st.success(f"🎉 Found Report **#{report_id}** created successfully!")
            st.info(f"💡 Please record your **Report ID: #{report_id}** to review incoming claims from owners in **My Reports**.")

            # Immediate AI/ML Matching against existing LOST items (FOUND -> LOST)
            st.markdown("---")
            st.markdown("### 🤖 Finding your suitable match...")
            new_report = db.get_report_by_id(report_id, include_private=False)
            lost_candidates = db.get_reports_for_matching("LOST")
            matches = matcher.find_matches_for_report(new_report, lost_candidates, threshold=0.25)

            if not matches:
                st.info("No matching lost reports found yet. When a student reports losing this item, our ML engine will alert them.")
            else:
                st.markdown(f"**Found {len(matches)} potential lost item owner(s):**")
                for m in matches:
                    cand = m["candidate"]
                    st.markdown(f"""
                    <div class="match-container">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-lost">LOST ITEM #{cand['id']}</span> &nbsp;
                                <span class="badge-match">{m['confidence_tier']}</span>
                                <h3 style="margin: 0.4rem 0 0.2rem 0; color:#1e1b4b;">{cand['item_name']}</h3>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size:0.8rem; color:#64748b; font-weight:600;">AI/ML PROBABILITY</div>
                                <div class="match-prob-high">{m['percentage']}%</div>
                            </div>
                        </div>
                        <p style="color:#334155; margin: 0.5rem 0;">{cand['description']}</p>
                        <p style="font-size:0.85rem; color:#475569; margin: 0;">
                            📍 <b>Reported Lost At:</b> {cand['location']} &nbsp;|&nbsp; 🎨 <b>Colour:</b> {cand['colour']} &nbsp;|&nbsp; 📅 <b>Date:</b> {cand['date']}
                        </p>
                        <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;" />
                        <div style="font-size:0.85rem; color:#1e293b;">
                            <b>Why this may match:</b>
                            <ul style="margin: 0.3rem 0; padding-left: 1.2rem;">
                                {''.join([f'<li>✓ {reason}</li>' for reason in m['explanations']])}
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_btn1, col_btn2 = st.columns([1, 2])
                    with col_btn1:
                        if st.button(f"🤝 Notify Owner (Report #{cand['id']})", key=f"btn_found_{cand['id']}", type="primary"):
                            ok, resp_msg = db.create_contact_request(report_id, cand['id'], "I have found an item matching your lost report.")
                            if ok:
                                st.success(resp_msg)
                            else:
                                st.warning(resp_msg)


# =========================================================
# PAGE 4: 🔎 FIND MATCHES
# =========================================================
elif menu_option == "🔎 Find Matches":
    st.markdown("## 🔎 Match Explorer")
    all_reports = db.get_all_reports()
    if not all_reports:
        st.warning("No reports registered yet in the database.")
    else:
        report_options = {
            f"[{r['report_type']}] #{r['id']} - {r['item_name']} ({r['location']})": r['id']
            for r in all_reports
        }
        selected_label = st.selectbox("Select a Report to Match:", list(report_options.keys()))
        selected_id = report_options[selected_label]

        threshold = 0.30

        target_report = db.get_report_by_id(selected_id, include_private=False)
        opposite_type = "FOUND" if target_report["report_type"] == "LOST" else "LOST"

        st.markdown(f"**Matching {target_report['report_type']} Item #{selected_id} against all {opposite_type} Items:**")
        
        matches = matcher.find_matches_by_id(selected_id, threshold=threshold)

        if not matches:
            st.info(f"No potential {opposite_type} matches found above {int(threshold*100)}% probability.")
        else:
            st.success(f"Identified **{len(matches)}** potential {opposite_type} match(es)!")
            for idx, m in enumerate(matches):
                cand = m["candidate"]
                badge_type = "badge-found" if opposite_type == "FOUND" else "badge-lost"
                
                with st.expander(f"Match #{idx+1}: {cand['item_name']} —Probability: {m['percentage']}% ({m['confidence_tier']})", expanded=True):
                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        if cand.get("image_path") and os.path.exists(cand["image_path"]):
                            st.image(cand["image_path"], caption=cand["item_name"], use_container_width=True)
                        else:
                            st.markdown("*(No image uploaded)*")

                    with col_info:
                        st.markdown(f"""
                        <span class="{badge_type}">{opposite_type} ITEM #{cand['id']}</span> &nbsp;
                        <span class="badge-match">{m['confidence_tier']}</span>
                        <h4 style="margin: 0.5rem 0 0.3rem 0;">{cand['item_name']}</h4>
                        <p style="color:#475569; font-size:0.92rem;">{cand['description']}</p>
                        <p style="font-size:0.85rem; color:#64748b;">
                            📍 <b>Location:</b> {cand['location']} &nbsp;|&nbsp; 🎨 <b>Colour:</b> {cand['colour']} &nbsp;|&nbsp; 📅 <b>Date:</b> {cand['date']}
                        </p>
                        """, unsafe_allow_html=True)

                        st.markdown("**Grounded Match Evidence:**")
                        for ex in m["explanations"]:
                            st.markdown(f"- ✓ {ex}")

                        st.markdown("---")
                        # Contact request flow
                        c_req1, c_req2 = st.columns([2, 1])
                        with c_req1:
                            req_msg = st.text_input("Message for contact request:", value="Hello, let's verify ownership and coordinate handover.", key=f"msg_match_{selected_id}_{cand['id']}")
                        with c_req2:
                            if st.button("🤝 Send Contact Request", key=f"btn_match_{selected_id}_{cand['id']}", type="primary"):
                                ok, resp = db.create_contact_request(selected_id, cand['id'], req_msg)
                                if ok:
                                    st.success(resp)
                                else:
                                    st.warning(resp)


# =========================================================
# PAGE 5: 📋 MY REPORTS & REQUESTS
# =========================================================
elif menu_option == "📋 My Reports & Requests":
    st.markdown("## 📋 Student Report Portal & Secure Contact Center")
    st.markdown("Lookup your active reports, review incoming contact requests, and manage secure handovers.")

    st.markdown("""
    <div class="privacy-notice">
        🔐 <b>Privacy Protection State Machine:</b> Contact numbers are masked by default. When you click <b>Accept Request</b>, phone numbers are mutually revealed so you can call or message to coordinate safe handover.
    </div>
    """, unsafe_allow_html=True)

    lookup_query = st.text_input("Enter your Report ID, Your Name, or Phone Number to load your dashboard:", placeholder="e.g. 1 or Aarav Sharma or 9876543210")

    if not lookup_query:
        st.info("👆 Please enter your Report ID, Name, or Phone Number to access your private reports.")
    else:
        user_reports = db.get_reports_by_user_query(lookup_query)
        if not user_reports:
            st.warning(f"No reports found matching query: '{lookup_query}'. Check your spelling or Report ID.")
        else:
            report_ids = [r["id"] for r in user_reports]
            requests_data = db.get_contact_requests_for_user(report_ids)

            tab_reports, tab_incoming, tab_outgoing = st.tabs([
                f"📦 My Reports ({len(user_reports)})",
                f"📥 Incoming Contact Requests ({len(requests_data['incoming'])})",
                f"📤 Outgoing Requests ({len(requests_data['outgoing'])})"
            ])

            # TAB 1: User's Reports
            with tab_reports:
                for rep in user_reports:
                    badge_class = "badge-lost" if rep["report_type"] == "LOST" else "badge-found"
                    with st.container():
                        st.markdown(f"""
                        <div class="report-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span class="{badge_class}">{rep['report_type']} ITEM #{rep['id']}</span> &nbsp;
                                    <span style="font-size:0.85rem; color:#64748b;">Reported: {rep['created_at']}</span>
                                </div>
                                <div><b>Status:</b> {rep['status']}</div>
                            </div>
                            <h3 style="margin: 0.5rem 0 0.3rem 0;">{rep['item_name']}</h3>
                            <p style="color:#475569; margin-bottom:0.5rem;">{rep['description']}</p>
                            <p style="font-size:0.85rem; color:#64748b;">
                                📍 <b>Location:</b> {rep['location']} &nbsp;|&nbsp; 🎨 <b>Colour:</b> {rep['colour']} &nbsp;|&nbsp; 📅 <b>Date:</b> {rep['date']}
                            </p>
                            <p style="font-size:0.85rem; color:#1e3a8a;">
                                🔑 <b>Your Secret Verification Detail:</b> {rep['identifying_details']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Delete report after explicit confirmation.
                        delete_confirm = st.checkbox(
                            "I confirm that I have recovered/handed over this item and want to permanently delete this report.",
                            key=f"delete_confirm_{rep['id']}"
                        )
                        if st.button(
                            f"🗑️ Delete Report #{rep['id']}",
                            key=f"delete_report_{rep['id']}",
                            disabled=not delete_confirm
                        ):
                            deleted, message = db.delete_report(rep["id"])
                            if deleted:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)

                        # Show quick matches for this report
                        opp_type = "FOUND" if rep["report_type"] == "LOST" else "LOST"
                        matches = matcher.find_matches_by_id(rep["id"], threshold=0.30)
                        if matches:
                            st.markdown(f"**⚡ AI identified {len(matches)} potential {opp_type} match(es) for this report:**")
                            for m in matches[:3]:
                                st.write(f"- **{m['candidate']['item_name']}** ({m['candidate']['location']}) — **{m['percentage']}% Match Probability**")
                        else:
                            st.caption("No potential matches detected yet for this report.")

            # TAB 2: Incoming Requests
            with tab_incoming:
                incoming_list = requests_data["incoming"]
                if not incoming_list:
                    st.info("No incoming contact requests yet.")
                else:
                    for req in incoming_list:
                        status_color = "#15803d" if req["status"] == "ACCEPTED" else ("#b91c1c" if req["status"] == "DECLINED" else "#b45309")
                        st.markdown(f"""
                        <div class="report-card" style="border-left: 4px solid {status_color};">
                            <div style="display:flex; justify-content:space-between;">
                                <b>Contact Request #{req['request_id']}</b>
                                <span style="font-weight:700; color:{status_color};">STATUS: {req['status']}</span>
                            </div>
                            <p style="margin:0.5rem 0 0.3rem 0; font-size:0.95rem;">
                                <b>{req['req_reporter_name']}</b> believes their {req['req_type']} item (<b>{req['req_item_name']}</b>) matches your report (<b>{req['tgt_item_name']}</b>).
                            </p>
                            <p style="font-style: italic; color:#475569; font-size:0.9rem;">
                                Message: "{req['requester_message']}"
                            </p>
                        """, unsafe_allow_html=True)

                        if req["status"] == "ACCEPTED":
                            st.success(f"📞 Contact Revealed: **{req['req_reporter_name']}** ({req['req_contact_number']})")
                        elif req["status"] == "DECLINED":
                            st.error("You declined this request. Contact details remain hidden.")
                        else:
                            st.info("🔒 Requester's contact number is hidden until you accept.")
                            btn_c1, btn_c2 = st.columns(2)
                            with btn_c1:
                                if st.button(f"✅ Accept Request #{req['request_id']}", key=f"acc_{req['request_id']}", type="primary"):
                                    db.accept_contact_request(req["request_id"])
                                    st.success("Request Accepted! Phone numbers are now mutually visible.")
                                    st.rerun()
                            with btn_c2:
                                if st.button(f"❌ Decline Request #{req['request_id']}", key=f"dec_{req['request_id']}"):
                                    db.decline_contact_request(req["request_id"])
                                    st.warning("Request Declined.")
                                    st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)

            # TAB 3: Outgoing Requests
            with tab_outgoing:
                outgoing_list = requests_data["outgoing"]
                if not outgoing_list:
                    st.info("You haven't sent any contact requests yet.")
                else:
                    for req in outgoing_list:
                        status_color = "#15803d" if req["status"] == "ACCEPTED" else ("#b91c1c" if req["status"] == "DECLINED" else "#b45309")
                        st.markdown(f"""
                        <div class="report-card" style="border-left: 4px solid {status_color};">
                            <div style="display:flex; justify-content:space-between;">
                                <b>Your Request #{req['request_id']}</b>
                                <span style="font-weight:700; color:{status_color};">STATUS: {req['status']}</span>
                            </div>
                            <p style="margin:0.5rem 0 0.3rem 0; font-size:0.95rem;">
                                You requested contact regarding {req['tgt_type']} item: <b>{req['tgt_item_name']}</b> reported by {req['tgt_reporter_name']}.
                            </p>
                            <p style="font-size:0.85rem; color:#64748b;">Requested on: {req['created_at']}</p>
                        """, unsafe_allow_html=True)

                        if req["status"] == "ACCEPTED":
                            st.success(f"🎉 Request Approved! Contact Phone: **{req['tgt_contact_number']}** ({req['tgt_reporter_name']})")
                        elif req["status"] == "DECLINED":
                            st.error("The owner/finder declined this contact request.")
                        else:
                            st.warning("⏳ Awaiting approval from the finder/owner. Phone number will be revealed once accepted.")

                        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE 6: 🌐 ALL PUBLIC REPORTS
# =========================================================
elif menu_option == "🌐 All Public Reports":
    st.markdown("## 🌐 Campus Public Registry")
    st.markdown("Browse all active lost and found reports. Contact details are strictly masked to protect student privacy.")

    st.markdown("""
    <div class="privacy-notice">
        🔒 <b>Zero Public Phone Numbers:</b> Contact details are private and will only be shared after a contact request is accepted. Please verify ownership using secret identifying details before handing over any item.
    </div>
    """, unsafe_allow_html=True)

    filter_c1, filter_c2, filter_c3 = st.columns([2, 1, 1])
    with filter_c1:
        search_kw = st.text_input("Search items, keywords, location...", placeholder="e.g. Milton, Calculator, Library, Blue")
    with filter_c2:
        type_filter = st.selectbox("Report Type", ["ALL", "LOST", "FOUND"])
    with filter_c3:
        cat_filter = st.selectbox("Category", ["All Categories", "Electronics", "Personal Belongings", "Daily Essentials", "Campus Specific", "Study Materials", "Bags & Luggage"])

    public_reports = db.get_all_reports(
        report_type=type_filter,
        search_query=search_kw,
        category=cat_filter
    )

    if not public_reports:
        st.info("No reports found matching your filter criteria.")
    else:
        st.caption(f"Displaying {len(public_reports)} public reports:")
        for r in public_reports:
            badge_class = "badge-lost" if r["report_type"] == "LOST" else "badge-found"
            
            with st.container():
                col_img, col_body = st.columns([1, 3])
                with col_img:
                    if r.get("image_path") and os.path.exists(r["image_path"]):
                        st.image(r["image_path"], caption=r["item_name"], use_container_width=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#f1f5f9; height:120px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#94a3b8; font-size:0.85rem;">
                            No Image Uploaded
                        </div>
                        """, unsafe_allow_html=True)

                with col_body:
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem;">
                        <span class="{badge_class}">{r['report_type']}</span> &nbsp;
                        <span style="font-size:0.85rem; color:#64748b;">Report #{r['id']} • {r['category']} • {r['date']}</span>
                        <h3 style="margin: 0.3rem 0; color:#0f172a;">{r['item_name']}</h3>
                        <p style="color:#475569; font-size:0.92rem; margin-bottom:0.4rem;">{r['description']}</p>
                        <p style="font-size:0.85rem; color:#64748b; margin:0;">
                            📍 <b>Location:</b> {r['location']} &nbsp;|&nbsp; 🎨 <b>Colour:</b> {r['colour']} &nbsp;|&nbsp; 👤 <b>Reported By:</b> {r['reporter_name']}
                        </p>
                        <p style="font-size:0.82rem; color:#059669; margin: 0.3rem 0 0 0;">
                            🔒 <b>Contact Number:</b> <i>Available through a private contact request.</i>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"🤝 Request Contact regarding Report #{r['id']}"):
                        my_id = st.number_input("Enter your own Report ID to connect with this item:", min_value=1, step=1, key=f"pub_req_id_{r['id']}")
                        pub_msg = st.text_input("Message:", value="Hi, I believe our reports correspond. Please review and accept contact.", key=f"pub_msg_{r['id']}")
                        if st.button("Send Contact Request", key=f"pub_btn_{r['id']}", type="primary"):
                            ok, resp = db.create_contact_request(int(my_id), r['id'], pub_msg)
                            if ok:
                                st.success(resp)
                            else:
                                st.warning(resp)
