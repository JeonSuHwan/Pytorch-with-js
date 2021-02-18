// import * as tf from '@tensorflow/tfjs@3.0.0';

const CANVAS_SIZE = 280;
const CANVAS_SCALE = 1.0;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

const fashion = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
];

// 모델 로드
async function load_model() {
    const model = await tf.loadLayersModel("../tfjsfModel/model.json");
    return model;
}

const m = load_model();

// 캔버스 초기 세팅
ctx.lineWidth = 12;
ctx.lineJoin = "round";
ctx.font = '28px sans-serif';
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillStyle = "#212121";
ctx.fillText("로딩중...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// 라인 색깔
ctx.strokeStyle = "#212121";

// 캔버스 비우기
function clearCanvas() {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    for (let i = 0; i < 10; i++) {
        const element = document.getElementById(`prediction-${i}`);
        element.className = "prediction-col";
        element.children[2].children[0].style.width = "0"
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

// 예측 함수
async function updatePrediction() {
    model = await tf.loadLayersModel("../tfjsfModel/model.json");
    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const data = imgData.data;
    // 이미지 데이터 전처리
    var arr = [];
    // 그레이 스케일링
    for (let i = 0; i < data.length; i += 4) {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = avg;
        arr.push(data[i]);
    }
    // 이미지 리사이징
    arr = arr.map(x => x / 255) // Normalization
    arr = tf.tensor3d(arr, [CANVAS_SIZE, CANVAS_SIZE, 1]);
    let img = tf.image.resizeBilinear(arr, [28, 28]);
    img = tf.squeeze(img, -1)
    img = tf.expandDims(img, 0);
    img = tf.expandDims(img, 0);
    // 결과 예측
    const prediction = model.predict(img);
    const pred = prediction.softmax();
    var result = pred.argMax(1).dataSync();
    var results = pred.dataSync();
    var result2 = pred.max(1).dataSync();
    document.getElementById("result").innerHTML = fashion[result[0]];

    for (let i = 0; i < results.length; i++) {
        const element = document.getElementById(`prediction-${i}`);
        element.children[2].children[0].style.width = `${(results[i] * 100)}%`;
        element.className =
            results[i] == result2 ?
            "prediction-col top-prediction" :
            "prediction-col";
    }

    model.dispose();
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

m.then(() => {
    canvas.addEventListener("mousedown", canvasMouseDown);
    canvas.addEventListener("mousemove", canvasMouseMove);
    document.body.addEventListener("mouseup", bodyMouseUp);
    document.body.addEventListener("mouseout", bodyMouseOut);
    clearButton.addEventListener("mousedown", clearCanvas);

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillText("옷을 그리세요!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
})