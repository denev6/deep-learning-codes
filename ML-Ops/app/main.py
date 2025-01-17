from fastapi import FastAPI, Request, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from static.function import convert_image, save_image, predict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

IMG_URL_PATH = "/img"
IMG_DIR = "static/img"
app.mount(IMG_URL_PATH, StaticFiles(directory=IMG_DIR), name="images")


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    """Upload an image using template"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/result/", response_class=HTMLResponse)
async def show_prediction(request: Request, file: UploadFile = File()):
    """Show prediction with template"""
    label, prob, img_tensor = await classify_img(file, verbose=True)
    file_name = save_image(img_tensor, IMG_DIR, file.filename)
    img_url = "/".join((IMG_URL_PATH, file_name))
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "img_src": img_url,
            "label": label,
            "prob": f"{prob * 100:.2f}%",
        },
    )


@app.post("/predict/")
async def get_prediction(file: UploadFile = File()):
    """Return prediction in JSON format"""
    label, prob = await classify_img(file)
    return {"label": label, "prob": prob}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "status_code": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code,
    )


async def is_valid_size(file: UploadFile):
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="첨부된 파일이 없습니다.",
        )
    if len(await file.read()) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="파일은 5MB 이하만 업로드 가능합니다.",
        )
    return file


async def is_valid_image(file: UploadFile):
    if not file.content_type.startswith("image"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미지 파일만 업로드 가능합니다.",
        )
    allowed_ext = ["jpg", "jpeg", "png"]
    file_ext = file.filename.rsplit(".", 1)[-1]
    if file_ext.lower() not in allowed_ext:
        allowed_ext_str = ", ".join(allowed_ext)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"이미지 확장자는 {allowed_ext_str}만 가능합니다.",
        )
    return file


async def classify_img(file: UploadFile, verbose=False):
    file = await is_valid_size(file)
    file = await is_valid_image(file)

    img_tensor = convert_image(file.file)
    label, prob = predict(img_tensor)
    if verbose:
        return label, prob, img_tensor
    return label, prob
