import qrcode

url = "https://ai-portfolio-manage.streamlit.app/"  
qr = qrcode.make(url)
qr.save("streamlit_app_qr.png")
print("✅ QR code saved as streamlit_app_qr.png")
