
(function () {
    var canvas = this.__canvas = new fabric.Canvas('c', {
        isDrawingMode: true
    });
  

    fabric.Object.prototype.transparentCorners = false;

    
    var recognizeEl = $('#recognize')[0],
        clearEl = $('#clear-canvas')[0];

    clearEl.onclick = function () {
        canvas.clear();
    };

    if (fabric.PatternBrush) {
        var vLinePatternBrush = new fabric.PatternBrush(canvas);
        vLinePatternBrush.getPatternSrc = function () {

            var patternCanvas = fabric.document.createElement('canvas');
            patternCanvas.width = patternCanvas.height = 10;
            var ctx = patternCanvas.getContext('2d');

            ctx.strokeStyle = this.color;
            ctx.lineWidth = 5;
            ctx.beginPath();
            ctx.moveTo(0, 5);
            ctx.lineTo(10, 5);
            ctx.closePath();
            ctx.stroke();

            return patternCanvas;
        };

        var hLinePatternBrush = new fabric.PatternBrush(canvas);
        hLinePatternBrush.getPatternSrc = function () {

            var patternCanvas = fabric.document.createElement('canvas');
            patternCanvas.width = patternCanvas.height = 10;
            var ctx = patternCanvas.getContext('2d');

            ctx.strokeStyle = this.color;
            ctx.lineWidth = 5;
            ctx.beginPath();
            ctx.moveTo(5, 0);
            ctx.lineTo(5, 10);
            ctx.closePath();
            ctx.stroke();

            return patternCanvas;
        };

        var squarePatternBrush = new fabric.PatternBrush(canvas);
        squarePatternBrush.getPatternSrc = function () {

            var squareWidth = 10, squareDistance = 2;

            var patternCanvas = fabric.document.createElement('canvas');
            patternCanvas.width = patternCanvas.height = squareWidth + squareDistance;
            var ctx = patternCanvas.getContext('2d');

            ctx.fillStyle = this.color;
            ctx.fillRect(0, 0, squareWidth, squareWidth);

            return patternCanvas;
        };

        var diamondPatternBrush = new fabric.PatternBrush(canvas);
        diamondPatternBrush.getPatternSrc = function () {

            var squareWidth = 10, squareDistance = 5;
            var patternCanvas = fabric.document.createElement('canvas');
            var rect = new fabric.Rect({
                width: squareWidth,
                height: squareWidth,
                angle: 45,
                fill: this.color
            });

            var canvasWidth = rect.getBoundingRectWidth();

            patternCanvas.width = patternCanvas.height = canvasWidth + squareDistance;
            rect.set({ left: canvasWidth / 2, top: canvasWidth / 2 });

            var ctx = patternCanvas.getContext('2d');
            rect.render(ctx);

            return patternCanvas;
        };
    }

    if (canvas.freeDrawingBrush) {
        canvas.freeDrawingBrush.color = "#000000";
        canvas.freeDrawingBrush.width = 10;
    }

    var getMnistData = function () {
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

    recognizeEl.onclick = async function () {
        const model = await tf.loadLayersModel('/MNIST/model.json');
        var mnistdat = getMnistData();
        let x = tf.tensor(mnistdat);
        x = tf.reshape(x, [1, 28, 28, 1]);
        let y = model.predict(x);
        var acc = y.max();
        y = y.argMax(1);
        var pred = y.dataSync()[0];
        acc = acc.dataSync()[0] * 100;

        document.getElementById("predValue").innerText = pred;
        document.getElementById("predAcc").innerText = acc;
    };
})();
