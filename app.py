import streamlit as st

from views import alerts, data, delivery, extract, match, model, overview, shortage
from views.common import inject_css

st.set_page_config(page_title="Procurement Verification and Risk", page_icon="📄", layout="wide")
inject_css()

pages = [
    st.Page(overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(extract.render, title="Extract documents", icon="📄", url_path="extract"),
    st.Page(match.render, title="Three-way match", icon="🔍", url_path="match"),
    st.Page(delivery.render, title="Delivery risk", icon="🚚", url_path="delivery"),
    st.Page(shortage.render, title="Shortage risk", icon="📦", url_path="shortage"),
    st.Page(alerts.render, title="Alerts", icon="🔔", url_path="alerts"),
    st.Page(model.render, title="Model performance", icon="📊", url_path="model"),
    st.Page(data.render, title="Data", icon="🗄️", url_path="data"),
]
st.navigation(pages).run()
