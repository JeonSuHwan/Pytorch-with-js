// import * as tf from '@tensorflow/tfjs@3.0.0';

const CANVAS_SIZE = 84;
const CANVAS_SCALE = 1.0;

const c1 = document.getElementById("c1");
const ctx1 = c1.getContext("2d");
const c2 = document.getElementById("c2");
const ctx2 = c2.getContext("2d");
const c3 = document.getElementById("c3");
const ctx3 = c3.getContext("2d");
const c4 = document.getElementById("c4");
const ctx4 = c4.getContext("2d");
const c5 = document.getElementById("c5");
const ctx5 = c5.getContext("2d");
const c6 = document.getElementById("c6");
const ctx6 = c6.getContext("2d");
const c7 = document.getElementById("c7");
const ctx7 = c7.getContext("2d");
const c8 = document.getElementById("c8");
const ctx8 = c8.getContext("2d");
const c9 = document.getElementById("c9");
const ctx9 = c9.getContext("2d");
const c10 = document.getElementById("c10");
const ctx10 = c10.getContext("2d");

const numbers = [
    "Zero", "One", "Two", "Three", "Four", "Five",
    "Six", "Seven", "Eight", "Nine"
];

const imageSources = [
    "img/zero.png",
    "img/one.png",
    "img/two.png",
    "img/three.png",
    "img/four.png",
    "img/five.png",
    "img/six.png",
    "img/seven.png",
    "img/eight.png",
    "img/nine.png"
];

const result = document.getElementById("result");

// 모델 로드
async function load_model() {
    m = tf.loadLayersModel("../tfjsModel/model.json");
    return m;
}
loadingModelPromise = load_model();

result.innerHTML = "Loading...";

function setImages() {
    var img = new Image();
    var img1 = new Image();
    var img2 = new Image();
    var img3 = new Image();
    var img4 = new Image();
    var img5 = new Image();
    var img6 = new Image();
    var img7 = new Image();
    var img8 = new Image();
    var img9 = new Image();

    img.src = imageSources[0];
    img1.src = imageSources[1];
    img2.src = imageSources[2];
    img3.src = imageSources[3];
    img4.src = imageSources[4];
    img5.src = imageSources[5];
    img6.src = imageSources[6];
    img7.src = imageSources[7];
    img8.src = imageSources[8];
    img9.src = imageSources[9];

    img.addEventListener('load', function() {
        ctx1.drawImage(img, 0, 0);
    }, false);

    img1.addEventListener('load', function() {
        ctx2.drawImage(img1, 0, 0);
    }, false);

    img2.addEventListener('load', function() {
        ctx3.drawImage(img2, 0, 0);
    }, false);

    img3.addEventListener('load', function() {
        ctx4.drawImage(img3, 0, 0);
    }, false);

    img4.addEventListener('load', function() {
        ctx5.drawImage(img4, 0, 0);
    }, false);

    img5.addEventListener('load', function() {
        ctx6.drawImage(img5, 0, 0);
    }, false);

    img6.addEventListener('load', function() {
        ctx7.drawImage(img6, 0, 0);
    }, false);

    img7.addEventListener('load', function() {
        ctx8.drawImage(img7, 0, 0);
    }, false);

    img8.addEventListener('load', function() {
        ctx9.drawImage(img8, 0, 0);
    }, false);

    img9.addEventListener('load', function() {
        ctx10.drawImage(img9, 0, 0);
    }, false);
}

function clickedCV0() { updatePrediction(ctx1); }

function clickedCV1() { updatePrediction(ctx2); }

function clickedCV2() { updatePrediction(ctx3); }

function clickedCV3() { updatePrediction(ctx4); }

function clickedCV4() { updatePrediction(ctx5); }

function clickedCV5() { updatePrediction(ctx6); }

function clickedCV6() { updatePrediction(ctx7); }

function clickedCV7() { updatePrediction(ctx8); }

function clickedCV8() { updatePrediction(ctx9); }

function clickedCV9() { updatePrediction(ctx10); }

// 예측함수 구현
async function updatePrediction(ctx) {
    const model = await tf.loadLayersModel("../tfjsModel/model.json");

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
    var r = prediction.argMax(1).dataSync();
    var rr = r[0];
    result.innerHTML = numbers[rr];
}

loadingModelPromise.then(() => {
    result.innerHTML = "";
    setImages();
    c1.addEventListener("mousedown", clickedCV0);
    c2.addEventListener("mousedown", clickedCV1);
    c3.addEventListener("mousedown", clickedCV2);
    c4.addEventListener("mousedown", clickedCV3);
    c5.addEventListener("mousedown", clickedCV4);
    c6.addEventListener("mousedown", clickedCV5);
    c7.addEventListener("mousedown", clickedCV6);
    c8.addEventListener("mousedown", clickedCV7);
    c9.addEventListener("mousedown", clickedCV8);
    c10.addEventListener("mousedown", clickedCV9);
})