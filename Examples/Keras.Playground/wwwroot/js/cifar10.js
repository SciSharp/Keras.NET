var labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
(function () {
    var recognizeEl = $('#btnpredict')[0];

    var getImageData = function () {
        var raww = canvas.width,
            rawh = canvas.height,
            mnistw = 28,
            mnisth = 28,
            relw = raww / mnistw,
            relh = rawh / mnisth;

        var rawdat = canvas.getContext().getImageData(0, 0, raww, rawh);

        var downsample = function (mnistx, mnisty) {
            var rawx = relw * mnistx;
            var rawy = relh * mnisty;

            var i = 1,
                avg = 0;
            for (var y = rawy; y < rawy + relh; y++) {
                for (var x = rawx; x < rawx + relw; x++ , i++) {
                    var rawidx = (y * raww + x) * 4;
                    var alpha = rawdat.data[rawidx + 3];
                    avg = avg + (1.0 / i) * (alpha - avg);
                }
            }

            return avg / 255.0;
        };

        var mnistdat = new Float32Array(mnistw * mnisth);
        for (var y = 0; y < mnisth; y++) {
            for (var x = 0; x < mnistw; x++) {
                var mnistidx = y * mnistw + x;
                mnistdat[mnistidx] = downsample(x, y);
            }
        }

        return Array.prototype.slice.call(mnistdat);
    };

    var loadImage = function (src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = src;
            img.onload = () => resolve(tf.fromPixels(img));
            img.onerror = (err) => reject(err);
        });
    };

    recognizeEl.onclick = async function () {
        debugger;
        const model = await tf.loadLayersModel('/CIFAR10/model.json');
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.width = 32;
        img.height = 32;
        img.src = document.getElementById("input_image").value;
        let x = tf.browser.fromPixels(img);
        //x = tf.image.resizeBilinear(x, [32, 32]);
        x = tf.reshape(x, [1, 32, 32, 3]);
        x = x.div(255);
        let y = model.predict(x);
        var acc = y.max();
        y = y.argMax(1);
        var pred = labels[y.dataSync()[0]];
        acc = acc.dataSync()[0] * 100;

        document.getElementById("predValue").innerText = pred;
        document.getElementById("predAcc").innerText = acc;
    };
})();
