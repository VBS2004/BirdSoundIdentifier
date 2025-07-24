import { useState, useEffect } from "react";
import { Mic, Upload, Play, Pause, Volume2, Loader2, Check, AlertCircle, Bird, Waves, Zap, Star, Camera, ExternalLink, Download, Heart } from "lucide-react";
import './index.css';

function PredictPage() {
  const [data, setData] = useState(null);
  const [val, setVal] = useState("");
  const [confidenceVal, setCV] = useState("");
  const [filename, setFilename] = useState("");
  const [choice, setChoice] = useState("Upload");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagesLoading, setImagesLoading] = useState(false);

  useEffect(() => {
    fetch("http://localhost:5000")
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        setData(data.message);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  }, []);

  const loadImages = async (imageUrls) => {
    setImagesLoading(true);
    try {
      const imageList = imageUrls.map((url, index) => ({
        id: index,
        url: `http://localhost:5000/${url}`,
        alt: `${val} - Image ${index + 1}`,
        loaded: false
      }));
      setImages(imageList);
    } catch (error) {
      console.error("Error loading images:", error);
    } finally {
      setImagesLoading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      alert("Please select or record a file first");
      return;
    }

    console.log("Uploading file:", file.name, "Size:", file.size, "Type:", file.type);
    setLoading(true);
    setShowResult(false);
    setImages([]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Backend response:", data);
      setVal(data.prediction);
      setCV(data.confidence);
      setShowResult(true);
      console.log("Image URLs:", data.image_urls);
      
      // Load images after successful prediction
      if (data.image_urls && data.image_urls.length > 0) {
        await loadImages(data.image_urls);
      } else {
        console.log("No image URLs received from backend");
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Error uploading file: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setFilename(uploadedFile.name);
      setAudioUrl(URL.createObjectURL(uploadedFile));
      setShowResult(false);
      setImages([]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      setFile(droppedFile);
      setFilename(droppedFile.name);
      setAudioUrl(URL.createObjectURL(droppedFile));
      setShowResult(false);
      setImages([]);
    }
  };

  const handleRecordingStop = (blob) => {
    console.log("Recording stopped:", blob);
    console.log("Blob type:", blob.type);
    console.log("Blob size:", blob.size);
    
    if (!blob || blob.size === 0) {
      alert("Recording failed - no audio data received");
      return;
    }
    
    const audioFile = new File([blob], `recording_${Date.now()}.wav`, { 
      type: blob.type || "audio/wav",
      lastModified: Date.now()
    });
    
    setFile(audioFile);
    setFilename(audioFile.name);
    setAudioUrl(URL.createObjectURL(audioFile));
    console.log("Audio file created:", audioFile);
  };

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  const handleImageLoad = (imageId) => {
    setImages(prev => prev.map(img => 
      img.id === imageId ? { ...img, loaded: true } : img
    ));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white overflow-hidden relative">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 h-60 bg-indigo-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse delay-500"></div>
      </div>

      {/* Header */}
      <header className="relative z-10 w-full backdrop-blur-md bg-black/20 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg">
              <Bird className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                BirdID
              </h1>
              <p className="text-xs text-gray-400">AI-Powered Bird Recognition</p>
            </div>
          </div>
          
          <nav className="hidden md:flex space-x-6">
            {[
              ['Dashboard', '/dashboard'],
              ['Species', '/species'], 
              ['Analytics', '/analytics'],
              ['Community', '/community'],
            ].map(([title, url]) => (
              <a 
                key={title}
                href={url} 
                className="px-4 py-2 rounded-full text-sm font-medium text-gray-300 hover:text-white hover:bg-white/10 transition-all duration-300 backdrop-blur-sm"
              >
                {title}
              </a>
            ))}
          </nav>

          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-xs text-green-400 hidden sm:block">AI Online</span>
          </div>
        </div>
      </header>

      <main className="relative z-10 flex-1 max-w-6xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h2 className="text-5xl md:text-7xl font-black mb-6 bg-gradient-to-r from-white via-blue-200 to-purple-200 bg-clip-text text-transparent leading-tight">
            Identify Birds
            <br />
            <span className="text-4xl md:text-6xl">Instantly</span>
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed">
            Upload an audio recording or record live to discover bird species with our advanced AI recognition system
          </p>
        </div>

        {/* Mode Selection */}
        <div className="flex justify-center mb-12">
          <div className="bg-black/30 backdrop-blur-md rounded-2xl p-2 border border-white/10">
            <div className="flex space-x-2">
              {[
                { value: "Upload", icon: Upload, label: "Upload Audio" },
                { value: "Record", icon: Mic, label: "Live Recording" }
              ].map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  onClick={() => setChoice(value)}
                  className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                    choice === value 
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg' 
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Interface */}
        <div className="max-w-2xl mx-auto">
          {choice === "Upload" ? (
            <div 
              className={`relative group cursor-pointer transition-all duration-300 ${
                dragOver ? 'scale-105' : ''
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className={`bg-gradient-to-br from-black/40 to-black/20 backdrop-blur-md border-2 border-dashed rounded-3xl p-12 text-center transition-all duration-300 ${
                dragOver 
                  ? 'border-blue-400 bg-blue-500/10' 
                  : 'border-gray-600 hover:border-gray-500 group-hover:bg-white/5'
              }`}>
                <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                  <Upload className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Drop your audio file here</h3>
                <p className="text-gray-400 mb-6">Or click to browse files</p>
                
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                
                <div className="inline-flex items-center space-x-2 px-6 py-3 bg-white/10 rounded-full text-sm font-medium backdrop-blur-sm">
                  <Zap className="w-4 h-4" />
                  <span>Supports MP3, WAV, M4A</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-gradient-to-br from-black/40 to-black/20 backdrop-blur-md border border-white/10 rounded-3xl p-12 text-center">
              <div className="w-24 h-24 mx-auto mb-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50 transition-all duration-300 cursor-pointer">
                <Mic className="w-10 h-10 text-white" />
              </div>
              
              <h3 className="text-2xl font-bold mb-4">Audio Recording</h3>
              <p className="text-gray-400 mb-6">Use an audio recording component here</p>
              <p className="text-sm text-gray-500">
                Note: Integrate with react-use-audio-recorder or similar library
              </p>
            </div>
          )}

          {/* File Info & Playback */}
          {file && (
            <div className="mt-8 bg-black/30 backdrop-blur-md rounded-2xl p-6 border border-white/10">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-blue-500 rounded-xl flex items-center justify-center">
                    <Check className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <p className="font-medium truncate max-w-xs">{filename}</p>
                    <p className="text-sm text-gray-400">Ready for analysis</p>
                  </div>
                </div>
                
                {audioUrl && (
                  <button
                    onClick={togglePlayback}
                    className="w-10 h-10 bg-white/10 hover:bg-white/20 rounded-xl flex items-center justify-center transition-colors duration-200"
                  >
                    {isPlaying ? (
                      <Pause className="w-5 h-5" />
                    ) : (
                      <Play className="w-5 h-5" />
                    )}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Analyze Button */}
          <div className="mt-8">
            <button
              onClick={handleSubmit}
              disabled={loading || !file}
              className={`w-full py-4 rounded-2xl font-bold text-lg transition-all duration-300 ${
                loading || !file
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50 transform hover:scale-[1.02]'
              }`}
            >
              {loading ? (
                <div className="flex items-center justify-center space-x-3">
                  <Loader2 className="w-6 h-6 animate-spin" />
                  <span>Analyzing Audio...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-3">
                  <Waves className="w-6 h-6" />
                  <span>Identify Bird Species</span>
                </div>
              )}
            </button>
          </div>

          {/* Results */}
          {showResult && val && (
            <div className="mt-12 bg-gradient-to-br from-green-500/20 to-blue-500/20 backdrop-blur-md border border-green-500/30 rounded-3xl p-8 animate-fadeIn">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-r from-green-400 to-blue-500 rounded-2xl flex items-center justify-center">
                  <Star className="w-8 h-8 text-white" />
                </div>
                
                <h3 className="text-3xl font-bold mb-4 bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                  Species Identified!
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <p className="text-gray-400 text-sm uppercase tracking-wide mb-2">Detected Species</p>
                    <p className="text-4xl font-black text-white">{val}</p>
                  </div>
                  
                  <div>
                    <p className="text-gray-400 text-sm uppercase tracking-wide mb-2">Confidence Score</p>
                    <div className="flex items-center justify-center space-x-3">
                      <div className="flex-1 bg-gray-700 rounded-full h-3 max-w-xs">
                        <div 
                          className="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full transition-all duration-1000"
                          style={{ width: confidenceVal }}
                        />
                      </div>
                      <span className="text-2xl font-bold text-green-400">{confidenceVal}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Image Gallery */}
        {showResult && images.length > 0 && (
          <div className="mt-16 max-w-6xl mx-auto">
            <div className="text-center mb-12">
              <div className="flex items-center justify-center space-x-3 mb-4">
                <Camera className="w-8 h-8 text-blue-400" />
                <h3 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  Species Gallery
                </h3>
              </div>
              <p className="text-gray-400">Visual references for {val}</p>
            </div>

            {imagesLoading ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                {[...Array(8)].map((_, i) => (
                  <div key={i} className="aspect-square bg-gradient-to-br from-gray-800 to-gray-700 rounded-2xl animate-pulse">
                    <div className="w-full h-full flex items-center justify-center">
                      <Loader2 className="w-8 h-8 text-gray-500 animate-spin" />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                {images.map((image, index) => (
                  <div
                    key={image.id}
                    className="group relative aspect-square bg-gradient-to-br from-black/40 to-black/20 backdrop-blur-md border border-white/10 rounded-2xl overflow-hidden cursor-pointer transform hover:scale-105 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/20"
                    onClick={() => setSelectedImage(image)}
                  >
                    <img
                      src={image.url}
                      alt={image.alt}
                      className={`w-full h-full object-cover transition-all duration-500 ${
                        image.loaded ? 'opacity-100' : 'opacity-0'
                      }`}
                      onLoad={() => handleImageLoad(image.id)}
                      onError={(e) => {
                        // Fallback to placeholder if image fails to load
                        e.target.style.display = 'none';
                      }}
                    />
                    
                    {!image.loaded && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
                      </div>
                    )}

                    {/* Overlay */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <div className="absolute bottom-4 left-4 right-4">
                        <p className="text-white font-semibold text-sm truncate">
                          {val} #{index + 1}
                        </p>
                        <div className="flex items-center justify-between mt-2">
                          <div className="flex space-x-2">
                            <button className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/40 transition-colors">
                              <Heart className="w-4 h-4" />
                            </button>
                            <button className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/40 transition-colors">
                              <Download className="w-4 h-4" />
                            </button>
                          </div>
                          <button className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/40 transition-colors">
                            <ExternalLink className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Image Modal */}
        {selectedImage && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
            <div className="relative max-w-4xl max-h-[90vh] bg-gradient-to-br from-black/60 to-black/40 backdrop-blur-md border border-white/20 rounded-3xl overflow-hidden">
              <button
                onClick={() => setSelectedImage(null)}
                className="absolute top-4 right-4 z-10 w-10 h-10 bg-black/50 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-black/70 transition-colors"
              >
                ✕
              </button>
              
              <img
                src={selectedImage.url}
                alt={selectedImage.alt}
                className="w-full h-auto max-h-[80vh] object-contain"
              />
              
              <div className="p-6 bg-gradient-to-t from-black/80 to-transparent">
                <h4 className="text-2xl font-bold text-white mb-2">{val}</h4>
                <p className="text-gray-300 mb-4">High-resolution reference image</p>
                <div className="flex space-x-3">
                  <button className="flex items-center space-x-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-xl hover:bg-white/20 transition-colors">
                    <Download className="w-4 h-4" />
                    <span>Download</span>
                  </button>
                  <button className="flex items-center space-x-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-xl hover:bg-white/20 transition-colors">
                    <ExternalLink className="w-4 h-4" />
                    <span>View Source</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="relative z-10 mt-20 border-t border-white/10 bg-black/20 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 py-8 text-center">
          <p className="text-gray-400">
            &copy; 2025 BirdID. Powered by advanced AI technology.
          </p>
        </div>
      </footer>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.6s ease-out;
        }
      `}</style>
    </div>
  );
}

export default PredictPage;