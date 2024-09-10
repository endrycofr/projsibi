Arsitektur AWS untuk Deploy Website SIBI dengan Skalabilitas dan Kinerja Tinggi
Untuk membuat arsitektur AWS guna mendukung deploy website SIBI dengan skalabilitas dan kinerja tinggi Arsitektur ini akan menggunakan beberapa layanan AWS seperti EC2, VPC,  Internet Gatewayserta Nginx sebagai reverse proxy dan Docker untuk isolasi aplikasi.
  ![Architecture Diagram](https://github.com/endrycofr/projsibi/blob/master/images/projsibi.png)
Berikut adalah langkah-langkah yang lebih disederhanakan dan disusun dengan baik untuk melakukan cloning dari GitHub, mempersiapkan kode untuk diisolasi menggunakan Docker, dan menjalankan aplikasi Streamlit dengan reverse proxy Nginx.

### 1. Clone Repositori dari GitHub
Clone repositori dari GitHub yang berisi kode aplikasi:
```bash
git clone -b master https://github.com/endrycofr/projsibi.git
```
Masuk ke direktori hasil clone:
```bash
cd projsibi
```

### 2. Membuat atau Mengedit File Dockerfile
Buat atau edit file **Dockerfile** untuk mengisolasi aplikasi di dalam container Docker:
```bash
nano Dockerfile
```
Isi file Dockerfile dengan konfigurasi berikut:
```dockerfile
# Menggunakan image Python terbaru
FROM python:3.11-slim-buster

# Mengekspos port yang akan digunakan oleh Streamlit
EXPOSE 8501

# Instalasi dependencies sistem yang dibutuhkan
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    software-properties-common \
    git \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, dan wheel untuk menghindari masalah
RUN pip3 install --upgrade pip setuptools wheel

# Set direktori kerja
WORKDIR /app

# Copy file requirements untuk instalasi dependencies
COPY requirements.txt .

# Instalasi dependencies aplikasi
RUN pip3 install -r requirements.txt --verbose

# Copy seluruh kode aplikasi ke container
COPY . .

# Set entry point untuk menjalankan aplikasi Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3. Build dan Jalankan Container Docker
Pastikan Docker sudah aktif:
```bash
sudo systemctl status docker
```
Lakukan build image Docker dari Dockerfile:
```bash
docker build -t mysibi-app .
```
Periksa apakah image telah berhasil di-build:
```bash
docker images
```
Jalankan container aplikasi:
```bash
docker run -d -p 8501:8501 mysibi-app
```
Periksa apakah container berjalan dengan benar:
```bash
docker ps
```

### 4. Menyiapkan Nginx sebagai Web Server
Periksa status Nginx:
```bash
sudo systemctl status nginx
```
Copy file statis dari aplikasi ke direktori web server Nginx:
```bash
sudo cp -r /home/ubuntu/projsibi /var/www/html/
```
Verifikasi direktori:
```bash
cd /var/www/html
ls
```

### 5. Konfigurasi Nginx untuk Reverse Proxy ke Streamlit App
Edit file konfigurasi Nginx:
```bash
sudo nano /etc/nginx/sites-available/default
```
Tambahkan konfigurasi berikut untuk membuat reverse proxy dan mengatur path untuk file statis:
```nginx
server {
    listen 80;
    server_name _;

    # Proxy untuk aplikasi Streamlit
    location / {
        proxy_pass http://localhost:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;

        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Log akses dan error untuk aplikasi
    access_log /var/log/nginx/projsibi_access.log;
    error_log /var/log/nginx/projsibi_error.log;

    # Pengaturan file statis
    root /var/www/html/projsibi;
    index index.html index.htm;

    location /static/ {
        alias /var/www/html/projsibi/static/;
    }

    location /media/ {
        alias /var/www/html/projsibi/media/;
    }

    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;

    location = /50x.html {
        root /usr/share/nginx/html;
    }
}
```

### 6. Verifikasi Konfigurasi Nginx
Periksa apakah konfigurasi Nginx valid:
```bash
sudo nginx -t
```
Jika tidak ada kesalahan, restart Nginx:
```bash
sudo systemctl restart nginx
```

### 7. Cek Aplikasi Streamlit
Gunakan **curl** untuk memeriksa apakah aplikasi Streamlit berjalan dengan benar:
```bash
curl -L http://127.0.0.1:8501
```

### 8. Akses Melalui IP Publik atau Privat
Setelah berhasil di-deploy, Anda bisa mengakses aplikasi melalui IP publik atau privat server, dengan mengetikkan alamat IP di browser:
```
http://<IP-ADDRESS>:8501
```

Dengan mengikuti langkah-langkah di atas, aplikasi berhasil dijalankan menggunakan Docker dan dilayani oleh Nginx sebagai reverse proxy.
