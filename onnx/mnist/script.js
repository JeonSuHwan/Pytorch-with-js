const CANVAS_SIZE = 280;
const CANVAS_SCALE = 1.0;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button")

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

//모델 로드
const sess = new onnx.InferenceSession();
const loadPromise = sess.loadModel("./mnist.onnx");

ctx.lineWidth = 28;
ctx.lineJoin = "round";
ctx.font = "28px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle"
ctx.fillStyle = "#212121"
ctx.fillText("로드중...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
ctx.strokeStyle = "#212121";

function clearCanvas() {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    for (let i = 0; i < 10; i++) {
        const element = document.getElementById(`prediction-${i}`);
        element.className = "prediction-col";
        element.children[0].children[0].style.height = "0";
    }
    document.getElementById("result").innerHTML = "";
}

// 선 그리기
function drawLine(fromX, fromY, toX, toY) {
    // fromX, fromY 에서 toX, toY로 라인을 그림
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.closePath();
    ctx.stroke();
    updatePrediction();
}

// 모델 예측 함수
async function updatePrediction() {
    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");

    const outputMap = await sess.run([input]);
    const outTensor = outputMap.values().next().value;
    const predictions = outTensor.data;
    const maxPrediction = Math.max(...predictions);

    for (let i = 0; i < predictions.length; i++) {
        const element = document.getElementById(`prediction-${i}`);
        element.children[0].children[0].style.height = `${predictions[i]*100}%`;
        element.className =
            predictions[i] === maxPrediction ?
            "prediction-col top-prediction" :
            "prediction-col";
        if (predictions[i] === maxPrediction) {
            document.getElementById("result").innerHTML = i;
        }
    }
}

function canvasMouseDown(event) {
    isMouseDown = true;
    if (hasIntroText) {
        clearCanvas();
        hasIntroText = false;
    }
    const x = event.offsetX / CANVAS_SCALE;
    const y = event.offsetY / CANVAS_SCALE;

    lastX = x + 0.001;
    lastY = y + 0.001;
    canvasMouseMove(event);
}

function canvasMouseMove(event) {
    const x = event.offsetX / CANVAS_SCALE;
    const y = event.offsetY / CANVAS_SCALE;
    if (isMouseDown) {
        drawLine(lastX, lastY, x, y);
    }
    lastX = x;
    lastY = y;
}

function bodyMouseUp() {
    isMouseDown = false;
}

function bodyMouseOut(event) {
    if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
        isMouseDown = false;
    }
}

loadPromise.then(() => {
    canvas.addEventListener("mousedown", canvasMouseDown);
    canvas.addEventListener("mousemove", canvasMouseMove);
    document.body.addEventListener("mouseup", bodyMouseUp);
    document.body.addEventListener("mouseout", bodyMouseOut);
    clearButton.addEventListener("mousedown", clearCanvas);

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillText("숫자를 그리세요!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
})