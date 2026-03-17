const API_BASE = window.location.origin;

class App {
    constructor() {
        this.stream = null;
        this.capturedImage = null;
        this.currentTaskId = null;
        this.selectedStation = null;
        this.selectedProduct = null;
        
        this.initElements();
        this.bindEvents();
        this.loadStations();
        this.loadProducts();
    }

    initElements() {
        this.elements = {
            stationSelect: document.getElementById('stationSelect'),
            productSelect: document.getElementById('productSelect'),
            refreshProducts: document.getElementById('refreshProducts'),
            videoPreview: document.getElementById('videoPreview'),
            captureCanvas: document.getElementById('captureCanvas'),
            imagePreview: document.getElementById('imagePreview'),
            captureHints: document.getElementById('captureHints'),
            startCamera: document.getElementById('startCamera'),
            captureImage: document.getElementById('captureImage'),
            retake: document.getElementById('retake'),
            fileInput: document.getElementById('fileInput'),
            uploadAndDetect: document.getElementById('uploadAndDetect'),
            resultSection: document.getElementById('resultSection'),
            resultStatus: document.getElementById('resultStatus'),
            resultDetails: document.getElementById('resultDetails'),
            similarityScore: document.getElementById('similarityScore'),
            errorList: document.getElementById('errorList'),
            historyList: document.getElementById('historyList'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            loadingText: document.getElementById('loadingText'),
        };
    }

    bindEvents() {
        this.elements.stationSelect.addEventListener('change', (e) => {
            this.selectedStation = e.target.value;
            this.updateCaptureHints();
        });

        this.elements.productSelect.addEventListener('change', (e) => {
            this.selectedProduct = e.target.value;
        });

        this.elements.refreshProducts.addEventListener('click', () => {
            this.loadProducts();
        });

        this.elements.startCamera.addEventListener('click', () => this.startCamera());
        this.elements.captureImage.addEventListener('click', () => this.captureImage());
        this.elements.retake.addEventListener('click', () => this.retake());

        this.elements.fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e);
        });

        this.elements.uploadAndDetect.addEventListener('click', () => this.uploadAndDetect());
    }

    async loadStations() {
        try {
            const response = await fetch(`${API_BASE}/api/stations`);
            const stations = await response.json();
            
            this.elements.stationSelect.innerHTML = '<option value="">请选择工位...</option>';
            for (const [id, station] of Object.entries(stations)) {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = station.name;
                this.elements.stationSelect.appendChild(option);
            }
        } catch (error) {
            console.error('加载工位失败:', error);
        }
    }

    async loadProducts() {
        try {
            const response = await fetch(`${API_BASE}/api/products`);
            const products = await response.json();
            
            this.elements.productSelect.innerHTML = '<option value="">请选择产品...</option>';
            products.forEach(product => {
                const option = document.createElement('option');
                option.value = product.id;
                option.textContent = `${product.name} (${product.code})`;
                this.elements.productSelect.appendChild(option);
            });
        } catch (error) {
            console.error('加载产品失败:', error);
        }
    }

    updateCaptureHints() {
        const hints = {
            '1': '请拍摄清晰的接线照片，确保线号和端子号可见',
            '2': '请从下往上逐层拍摄，每层单独拍照',
            '3': '请确保排线插头完全入位',
            '4': '请拍摄PLC模块全貌，确保型号清晰',
            '5': '请检查短接线连接是否正确',
            '6': '请检查短接片安装是否到位',
        };
        
        const hint = hints[this.selectedStation] || '请拍摄清晰的接线照片';
        this.elements.captureHints.querySelector('p').textContent = hint;
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            });
            this.elements.videoPreview.srcObject = this.stream;
            this.elements.videoPreview.style.display = 'block';
            this.elements.imagePreview.style.display = 'none';
            this.elements.startCamera.style.display = 'none';
            this.elements.captureImage.disabled = false;
            this.elements.captureHints.querySelector('p').textContent = '相机已开启，取景构图后点击拍照';
        } catch (error) {
            alert('无法开启相机: ' + error.message);
        }
    }

    captureImage() {
        const canvas = this.elements.captureCanvas;
        const video = this.elements.videoPreview;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        this.capturedImage = canvas.toDataURL('image/jpeg', 0.9);
        
        this.stopCamera();
        
        this.elements.imagePreview.src = this.capturedImage;
        this.elements.imagePreview.style.display = 'block';
        this.elements.videoPreview.style.display = 'none';
        
        this.elements.captureImage.style.display = 'none';
        this.elements.retake.style.display = 'inline-block';
        this.elements.startCamera.style.display = 'none';
        
        this.checkReadyToDetect();
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }

    retake() {
        this.capturedImage = null;
        this.elements.imagePreview.style.display = 'none';
        this.elements.captureImage.style.display = 'inline-block';
        this.elements.retake.style.display = 'none';
        this.elements.startCamera.style.display = 'inline-block';
        this.elements.captureImage.disabled = true;
        this.elements.uploadAndDetect.disabled = true;
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            this.capturedImage = e.target.result;
            this.elements.imagePreview.src = this.capturedImage;
            this.elements.imagePreview.style.display = 'block';
            this.elements.videoPreview.style.display = 'none';
            this.elements.captureImage.style.display = 'none';
            this.elements.retake.style.display = 'inline-block';
            this.elements.startCamera.style.display = 'none';
            this.checkReadyToDetect();
        };
        reader.readAsDataURL(file);
    }

    checkReadyToDetect() {
        const ready = this.capturedImage && this.selectedStation && this.selectedProduct;
        this.elements.uploadAndDetect.disabled = !ready;
    }

    async uploadAndDetect() {
        if (!this.capturedImage || !this.selectedStation || !this.selectedProduct) {
            alert('请完成所有选择并拍摄/选择图片');
            return;
        }

        this.showLoading('正在创建任务...');

        try {
            const createResponse = await fetch(`${API_BASE}/api/tasks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    product_id: parseInt(this.selectedProduct),
                    station_id: parseInt(this.selectedStation),
                    layer: 1
                })
            });

            if (!createResponse.ok) throw new Error('创建任务失败');
            
            const task = await createResponse.json();
            this.currentTaskId = task.id;

            this.showLoading('正在上传图片...');
            
            const formData = new FormData();
            const blob = this.dataURLToBlob(this.capturedImage);
            formData.append('file', blob, 'capture.jpg');

            const uploadResponse = await fetch(`${API_BASE}/api/tasks/${this.currentTaskId}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) throw new Error('上传图片失败');

            this.showLoading('正在检测中...');

            const detectResponse = await fetch(`${API_BASE}/api/tasks/${this.currentTaskId}/detect`, {
                method: 'POST'
            });

            if (!detectResponse.ok) throw new Error('检测失败');

            const result = await detectResponse.json();
            this.displayResult(result);
            this.loadHistory();

        } catch (error) {
            alert('操作失败: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayResult(result) {
        this.elements.resultSection.style.display = 'block';
        
        const isPass = result.overall_result === 'pass';
        const statusBadge = this.elements.resultStatus.querySelector('.status-badge') || document.createElement('span');
        statusBadge.className = `status-badge ${isPass ? 'pass' : 'fail'}`;
        statusBadge.textContent = isPass ? '合格' : '不合格';
        
        if (!this.elements.resultStatus.querySelector('.status-badge')) {
            this.elements.resultStatus.appendChild(statusBadge);
        }

        this.elements.similarityScore.textContent = ((result.similarity_score || 0) * 100).toFixed(1) + '%';

        const errors = result.errors || [];
        this.elements.errorList.innerHTML = '';
        
        if (errors.length === 0) {
            this.elements.errorList.innerHTML = '<p style="color: #10b981; text-align: center;">未发现异常</p>';
        } else {
            errors.forEach(error => {
                const item = document.createElement('div');
                item.className = 'error-item';
                item.innerHTML = `
                    <div class="module">${error.module}</div>
                    <div class="message">${error.message}</div>
                `;
                this.elements.errorList.appendChild(item);
            });
        }

        this.elements.resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    async loadHistory() {
        try {
            const response = await fetch(`${API_BASE}/api/tasks?limit=10`);
            const tasks = await response.json();

            if (tasks.length === 0) {
                this.elements.historyList.innerHTML = '<p class="empty-hint">暂无检测记录</p>';
                return;
            }

            this.elements.historyList.innerHTML = '';
            
            tasks.forEach(task => {
                const item = document.createElement('div');
                item.className = 'history-item';
                const date = task.created_at ? new Date(task.created_at).toLocaleString() : '未知';
                const isPass = task.overall_result === 'pass';
                
                item.innerHTML = `
                    <div class="info">
                        <div class="station">工位 ${task.station_id}</div>
                        <div class="time">${date}</div>
                    </div>
                    <span class="status ${isPass ? 'pass' : 'fail'}">${isPass ? '合格' : '不合格'}</span>
                `;
                this.elements.historyList.appendChild(item);
            });
        } catch (error) {
            console.error('加载历史失败:', error);
        }
    }

    dataURLToBlob(dataURL) {
        const arr = dataURL.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        
        return new Blob([u8arr], { type: mime });
    }

    showLoading(text) {
        this.elements.loadingText.textContent = text;
        this.elements.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.elements.loadingOverlay.style.display = 'none';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new App();
});