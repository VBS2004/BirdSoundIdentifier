import { useState, useEffect, useRef } from "react";
import { Mic, Upload, Play, Pause, Loader2, Check, Bird, Waves, Zap, Star, Camera, ExternalLink, Download, Heart, Square, RotateCcw } from "lucide-react";
import "./index.css";
function PredictPage() {
  const [data, setData] = useState(null);
  const [showDownloadNotification, setShowDownloadNotification] = useState(false);
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

  // Audio recording states
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingError, setRecordingError] = useState("");
  
  // Audio recording refs
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const animationRef = useRef(null);

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

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

const handleDownload=async(image)=>{
  try{
    const url=image.originalUrl || image.largeUrl || image.mediumUrl || image.smallUrl;
    const response=await fetch(url);
    if(!response.ok) throw new Error("Failed to fetch image");
    setShowDownloadNotification(true);
    const blob=await response.blob();
    const blobUrl=window.URL.createObjectURL(blob);//Temprorary URL for the blob
    const link=document.createElement("a");
    link.href= blobUrl;//Setting url
    link.download=`${val}_${image.id}.${url.split('.').slice(-1)[0]}`;//File name format Tells the browser to download instead of navigating
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(blobUrl);
    setTimeout(() => setShowDownloadNotification(false), 2000);
  }
  catch(error){
    console.error("Download error:", error);
    alert("Failed to download image: " + error.message);
    setShowDownloadNotification(false);
  }
}

const handleViewSource=(image)=>{
  const sourceUrl=`https://www.pexels.com/photo/${image.pexelsId}/`;
  window.open(sourceUrl, "_blank"); 
}

const loadImages = async (imageUrlsData) => {
  setImagesLoading(true);
  try {
    // imageUrlsData is now an array of objects with different size URLs
    const imageList = imageUrlsData.map((imageData, index) => ({
      id: index,
      url: imageData.medium, // Use medium size, or choose another size
      originalUrl: imageData.original,
      largeUrl: imageData.large,
      smallUrl: imageData.small,
      photographer: imageData.photographer,
      alt: imageData.alt || `${val} - Image ${index + 1}`,
      pexelsId: imageData.id,
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

  // Audio recording functions
  const startRecording = async () => {
    try {
      setRecordingError("");
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        }
      });
      
      streamRef.current = stream;
      chunksRef.current = [];

      // Set up audio context for visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      analyserRef.current.fftSize = 256;
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);

      // Audio level monitoring
      const updateAudioLevel = () => {
        if (analyserRef.current && isRecording && !isPaused) {
          analyserRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          setAudioLevel(Math.min(100, (average / 255) * 100));
          animationRef.current = requestAnimationFrame(updateAudioLevel);
        }
      };

      // Set up MediaRecorder
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4'
      });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { 
          type: mediaRecorderRef.current.mimeType 
        });
        
        if (blob.size > 0) {
          const audioFile = new File([blob], `recording_${Date.now()}.webm`, { 
            type: blob.type,
            lastModified: Date.now()
          });
          
          setFile(audioFile);
          setFilename(audioFile.name);
          setAudioUrl(URL.createObjectURL(audioFile));
          setShowResult(false);
          setImages([]);
        }

        // Cleanup
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
        if (audioContextRef.current) {
          audioContextRef.current.close();
        }
      };

      // Start recording
      mediaRecorderRef.current.start(100); // Collect data every 100ms
      setIsRecording(true);
      setIsPaused(false);
      setRecordingTime(0);
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      // Start audio level monitoring
      updateAudioLevel();
      
    } catch (error) {
      console.error("Error starting recording:", error);
      setRecordingError("Microphone access denied or not available");
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        setIsPaused(false);
        // Resume timer
        timerRef.current = setInterval(() => {
          setRecordingTime(prev => prev + 1);
        }, 1000);
      } else {
        mediaRecorderRef.current.pause();
        setIsPaused(true);
        // Pause timer
        if (timerRef.current) {
          clearInterval(timerRef.current);
        }
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsPaused(false);
      setAudioLevel(0);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
  };

  const resetRecording = () => {
    stopRecording();
    setRecordingTime(0);
    setFile(null);
    setFilename("");
    setAudioUrl(null);
    setShowResult(false);
    setImages([]);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
      {showDownloadNotification && (
    <div className="fixed top-4 right-4 z-50 px-4 py-2 bg-green-500/90 text-white rounded-xl shadow-lg backdrop-blur-sm animate-slideInOut">
      Download Started
    </div>
  )}
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 h-60 bg-indigo-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse delay-500"></div>
      </div>


      <main className="relative z-10 flex-1 max-w-6xl mx-auto px-6 py-12 pt-20">
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
                  : 'border-gray-600 hover:border-gray-500'
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
            // Audio Recording Component
            <div className="bg-gradient-to-br from-black/40 to-black/20 backdrop-blur-md border border-white/10 rounded-3xl p-12 text-center">
              {/* Recording Visualizer */}
              <div className="relative mb-8">
                <div className={`w-32 h-32 mx-auto rounded-full flex items-center justify-center transition-all duration-300 ${
                  isRecording 
                    ? 'bg-gradient-to-r from-red-500 to-pink-600 shadow-lg shadow-red-500/50 animate-pulse' 
                    : 'bg-gradient-to-r from-blue-500 to-purple-600 shadow-lg shadow-purple-500/30'
                }`}>
                  <Mic className="w-12 h-12 text-white" />
                  
                  {/* Audio level indicator */}
                  {isRecording && (
                    <div 
                      className="absolute inset-0 rounded-full border-4 border-white/30 transition-all duration-100"
                      style={{ 
                        transform: `scale(${1 + (audioLevel / 100) * 0.3})`,
                        opacity: audioLevel / 100 
                      }}
                    />
                  )}
                </div>
                
                {/* Recording timer */}
                {(isRecording || recordingTime > 0) && (
                  <div className="mt-4">
                    <div className="text-3xl font-mono font-bold text-white mb-2">
                      {formatTime(recordingTime)}
                    </div>
                    <div className="flex items-center justify-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`} />
                      <span className="text-sm text-gray-400">
                        {isPaused ? 'Paused' : isRecording ? 'Recording' : 'Stopped'}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Recording Error */}
              {recordingError && (
                <div className="mb-6 p-4 bg-red-500/20 border border-red-500/30 rounded-2xl">
                  <p className="text-red-300 text-sm">{recordingError}</p>
                </div>
              )}

              {/* Recording Controls */}
              <div className="flex items-center justify-center space-x-4 mb-6">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    className="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 rounded-2xl font-bold text-white shadow-lg shadow-red-500/30 hover:shadow-red-500/50 transform hover:scale-105 transition-all duration-300"
                  >
                    <Mic className="w-5 h-5" />
                    <span>Start Recording</span>
                  </button>
                ) : (
                  <>
                    <button
                      onClick={pauseRecording}
                      className="flex items-center space-x-2 px-6 py-3 bg-yellow-500 hover:bg-yellow-600 rounded-xl font-medium text-white transition-colors duration-200"
                    >
                      {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                      <span>{isPaused ? 'Resume' : 'Pause'}</span>
                    </button>
                    
                    <button
                      onClick={stopRecording}
                      className="flex items-center space-x-2 px-6 py-3 bg-red-500 hover:bg-red-600 rounded-xl font-medium text-white transition-colors duration-200"
                    >
                      <Square className="w-4 h-4" />
                      <span>Stop</span>
                    </button>
                  </>
                )}

                {/* Reset button (only show if there's a recording) */}
                {(file || recordingTime > 0) && (
                  <button
                    onClick={resetRecording}
                    className="flex items-center space-x-2 px-6 py-3 bg-gray-600 hover:bg-gray-700 rounded-xl font-medium text-white transition-colors duration-200"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>Reset</span>
                  </button>
                )}
              </div>

              {/* Instructions */}
              <div className="space-y-2 text-sm text-gray-400">
                <p>Click "Start Recording" to begin capturing bird sounds</p>
                <p>Make sure your microphone is enabled and positioned to capture audio clearly</p>
                <p>Recording will automatically stop after 60 seconds or when you click "Stop"</p>
              </div>
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
                          style={{ width: `${confidenceVal}%` }}
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
                    {console.log("Rendering image:", image.url)}
                    <img
                      src={image.url}
                      alt={image.alt}
                      className={`w-full h-full object-cover transition-all duration-500 ${
                        image.loaded ? 'opacity-100' : 'opacity-0'
                      }`}
                      onLoad={() => handleImageLoad(image.id)}
                      onError={(e) => {
                        console.error("Failed to load image:", image.url);
                        // Try fallback to small size if medium fails
                        if (image.smallUrl && e.target.src !== image.smallUrl) {
                          e.target.src = image.smallUrl;
                        } else {
                          e.target.style.display = 'none';
                        }
                      }}
                    />
                    
                    {!image.loaded && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
                      </div>
                    )}

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
                            <button onClick={()=>handleDownload(image)} className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/40 transition-colors">
                              <Download className="w-4 h-4" />
                            </button>
                          </div>
                          <button onClick={()=>handleViewSource(image)} className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/40 transition-colors">
                            <ExternalLink  className="w-4 h-4" />
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
                  <button onClick={()=>handleDownload(selectedImage)} className="flex items-center space-x-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-xl hover:bg-white/20 transition-colors">
                    <Download className="w-4 h-4" />
                    <span>Download</span>
                  </button>
                  <button onClick={()=>handleViewSource(selectedImage)} className="flex items-center space-x-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-xl hover:bg-white/20 transition-colors">
                    <ExternalLink  className="w-4 h-4" />
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