# -------------------- UPI FRAUD DETECTION - WEB APP --------------------
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys

# ✅ Import fraud-checking function
sys.path.append(os.path.dirname(__file__))
from infer import check_transaction as predict_transaction

# ------------------------------------------------------------
app = FastAPI()

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templets")

# Mount static files & templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ------------------------------------------------------------
# 1️⃣ HOME PAGE
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Homepage (index.html)"""
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------------------------------------------------
# 2️⃣ PREDICT ENDPOINT
@app.post("/predict")
async def predict(
    request: Request,
    user_id: str = Form(...),
    device_id: str = Form(...),
    merchant_id: str = Form(...),
    amount: float = Form(...)
):
    """
    Handles fraud detection.
    If fraud is detected → show confirmation page.
    Otherwise → go directly to PIN page.
    """
    result = predict_transaction(user_id, merchant_id, device_id, amount)
    probability = result.get("probability", 0)

    if probability > 0.5:
        # 🚨 Fraud detected → ask user confirmation
        return templates.TemplateResponse(
            "confirm.html",
            {
                "request": request,
                "user_id": user_id,
                "merchant_id": merchant_id,
                "device_id": device_id,
                "amount": amount,
                "note": result.get("note", "⚠️ Suspicious transaction detected!"),
            },
        )
    else:
        # ✅ Safe transaction → proceed to PIN page
        redirect_url = f"/pin?amount={amount}"
        return RedirectResponse(url=redirect_url, status_code=303)


# ------------------------------------------------------------
# 3️⃣ CONFIRMATION PAGE
@app.post("/confirm")
async def confirm_decision(
    request: Request,
    user_id: str = Form(...),
    merchant_id: str = Form(...),
    device_id: str = Form(...),
    amount: float = Form(...),
    decision: str = Form(...)
):
    """User confirms whether to proceed after fraud alert."""
    if decision == "yes":
        # Continue to PIN page
        redirect_url = f"/pin?amount={amount}"
        return RedirectResponse(url=redirect_url, status_code=303)
    else:
        # Back to home
        return RedirectResponse(url="/", status_code=303)


# ------------------------------------------------------------
# 4️⃣ PIN ENTRY PAGE
@app.get("/pin", response_class=HTMLResponse)
async def pin_page(request: Request, amount: float = 0.0):
    """Show PIN entry page with the transaction amount."""
    return templates.TemplateResponse(
        "pin.html",
        {"request": request, "amount": amount},
    ) 


# ------------------------------------------------------------
# 5️⃣ SUCCESS PAGE
@app.get("/success", response_class=HTMLResponse)
async def success_page(request: Request, amount: float = 0.0):
    """Show payment success confirmation."""
    return templates.TemplateResponse(
        "success.html",
        {"request": request, "amount": amount},
    )


# ------------------------------------------------------------
# Run locally (for testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
