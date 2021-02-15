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

var fashion = ["T-shirt", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Angkle boot"
];

const imageSources = [
    "img/t-shirt.jpg",
    "img/trouser.jpg",
    "img/pullover.jpg",
    "img/dress.jpg",
    "img/coat.jpg",
    "img/sandal.jpg",
    "img/shirt.jpg",
    "img/sneaker.jpg",
    "img/bag.jpg",
    "img/ankle-boot.jpg"
]

const result = document.getElementById("result");
const percent = document.getElementById("percent");

// 모델 로드
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./newfmnist.onnx");

result.innerHTML = "Loading...";

function setImages() {
    var img = new Image();
    img.addEventListener('load', function() {
        ctx1.drawImage(img, 0, 0);
    }, false);
    img.src = imageSources[0];

    var img2 = new Image();
    img2.addEventListener('load', function() {
        ctx2.drawImage(img2, 0, 0);
    }, false);
    img2.src = imageSources[1];

    var img3 = new Image();
    img3.addEventListener('load', function() {
        ctx3.drawImage(img3, 0, 0);
    }, false);
    img3.src = imageSources[2];

    var img4 = new Image();
    img4.addEventListener('load', function() {
        ctx4.drawImage(img4, 0, 0);
    }, false);
    img4.src = imageSources[3];

    var img5 = new Image();
    img5.addEventListener('load', function() {
        ctx5.drawImage(img5, 0, 0);
    }, false);
    img5.src = imageSources[4];

    var img6 = new Image();
    img6.addEventListener('load', function() {
        ctx6.drawImage(img6, 0, 0);
    }, false);
    img6.src = imageSources[5];

    var img7 = new Image();
    img7.addEventListener('load', function() {
        ctx7.drawImage(img7, 0, 0);
    }, false);
    img7.src = imageSources[6];

    var img8 = new Image();
    img8.addEventListener('load', function() {
        ctx8.drawImage(img8, 0, 0);
    }, false);
    img8.src = imageSources[7];

    var img9 = new Image();
    img9.addEventListener('load', function() {
        ctx9.drawImage(img9, 0, 0);
    }, false);
    img9.src = imageSources[8];

    var img10 = new Image();
    img10.addEventListener('load', function() {
        ctx10.drawImage(img10, 0, 0);
    }, false);
    img10.src = imageSources[9];
}

function clickedCanvas1() {
    updatePrediction(ctx1);
}

function clickedCanvas2() {
    updatePrediction(ctx2);
}

function clickedCanvas3() {
    updatePrediction(ctx3);
}

function clickedCanvas4() {
    updatePrediction(ctx4);
}

function clickedCanvas5() {
    updatePrediction(ctx5);
}

function clickedCanvas6() {
    updatePrediction(ctx6);
}

function clickedCanvas7() {
    updatePrediction(ctx7);
}

function clickedCanvas8() {
    updatePrediction(ctx8);
}

function clickedCanvas9() {
    updatePrediction(ctx9);
}

function clickedCanvas10() {
    updatePrediction(ctx10);
}

async function updatePrediction(ctx) {
    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    // const data = imgData.data;
    // // // 그레이 스케일링
    // for (var i = 0; i < data.length; i += 4) {
    //     var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
    //     data[i] = avg;
    //     data[i + 1] = avg;
    //     data[i + 2] = avg;
    // }

    // arr = new Float32Array(data);
    // for (var i = 0; i < arr.length; i += 4) {
    //     arr[i] = arr[i] / 255.0;
    //     arr[i + 1] = arr[i + 1] / 255.0;
    //     arr[i + 2] = arr[i + 2] / 255.0;
    // }

    const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");

    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data;
    console.log(predictions);
    const maxPrediction = Math.max(...predictions);

    var max = -1;
    for (let i = 0; i < predictions.length; i++) {
        if (max < predictions[i]) max = i;
    }
    result.innerHTML = fashion[max];
    percent.innerHTML = maxPrediction.toFixed(2) * 100 + "%";
}

loadingModelPromise.then(() => {
    result.innerHTML = "";
    setImages();
    c1.addEventListener("mousedown", clickedCanvas1);
    c2.addEventListener("mousedown", clickedCanvas2);
    c3.addEventListener("mousedown", clickedCanvas3);
    c4.addEventListener("mousedown", clickedCanvas4);
    c5.addEventListener("mousedown", clickedCanvas5);
    c6.addEventListener("mousedown", clickedCanvas6);
    c7.addEventListener("mousedown", clickedCanvas7);
    c8.addEventListener("mousedown", clickedCanvas8);
    c9.addEventListener("mousedown", clickedCanvas9);
    c10.addEventListener("mousedown", clickedCanvas10);
})