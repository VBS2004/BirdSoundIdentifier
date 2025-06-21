import { useState, useEffect } from "react";
import axios from "axios";
import './index.css';
import AudioRecorder from "react-use-audio-recorder";
import "react-use-audio-recorder/dist/index.css";

function App() {
  const [data, setData] = useState("");
  const [val, setVal] = useState("Upload audio file to predict");
  const [confidenceVal, setCV] = useState("");
  const [filename, setFilename] = useState("No file Uploaded");
  const [choice, setChoice] = useState("Upload"); // Fixed: Match the radio value
  const [file, setFile] = useState(null);

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

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!file) {
      alert("Please select or record a file first");
      return;
    }

    console.log("Uploading file:", file.name, "Size:", file.size, "Type:", file.type);

    const formData = new FormData();
    formData.append("file", file);

    // Add headers to ensure proper file handling
    const config = {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 30000, // 30 second timeout
    };

    try {
      const response = await axios.post("http://localhost:5000/upload", formData, config);
      console.log(response.data.message);
      setVal(response.data.prediction);
      setCV(response.data.confidence);
      alert("File uploaded successfully");
    } catch (error) {
      console.error("Upload error:", error);
      alert("Error uploading file: " + (error.response?.data?.message || error.message));
    }
  };

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setFilename(uploadedFile.name);
    }
  };

  const handleRecordingStop = (blob) => {
    console.log("Recording stopped:", blob);
    console.log("Blob type:", blob.type);
    console.log("Blob size:", blob.size);
    
    // Ensure we have a valid audio blob
    if (!blob || blob.size === 0) {
      alert("Recording failed - no audio data received");
      return;
    }
    
    // Convert blob to File object with proper MIME type
    const audioFile = new File([blob], `recording_${Date.now()}.wav`, { 
      type: blob.type || "audio/wav",
      lastModified: Date.now()
    });
    
    setFile(audioFile);
    setFilename(audioFile.name);
    console.log("Audio file created:", audioFile);
  };

  return (
    
    <div className="min-h-screen flex flex-col items-center justify-start bg-white">
      <header className="w-full flex items-center hover:bg-purple-400 mb-5">
        <div className="flex items-center">
          <img src="/logo.png" alt="BirdID Logo" className="h-20 max-w-full"/>
          <h1 className="ml-2 text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-violet-300 to-fuchsia-400">
            BirdID
          </h1>
        </div>
        
        {/* Navigation */}
        <nav className="ml-auto flex space-x-6 justify-between px-6">
          {[
            ['Home', '/dashboard'],
            ['Team', '/team'], 
            ['Projects', '/projects'],
            ['Reports', '/reports'],
          ].map(([title, url]) => (
            <a className="rw-cta-text px-4 pt-[10px] pb-[11px] rounded-full inline-flex transition-all duration-200 items-center justify-center text-purple-600 border border-offBlack hover:border-black bg-offBlack hover:text-black min-[1600px]:border-offBlack min-[1600px]:text-offBlack opacity-0 lg:opacity-100"
              key={title}
              href={url} 
            >
              {title}
            </a>
          ))}
        </nav>
      </header>
      <main className="flex-grow flex flex-col items-center">
          <div className="inline">
            <span className="ms-8 text-lg">
              <input 
                className="accent-violet-600 w-4 h-4" 
                checked={choice === "Record"} 
                type="radio" 
                id="Record" 
                name="type" 
                value="Record"  
                onChange={(e) => setChoice(e.target.value)}
              />
              <label htmlFor="Record" className="ml-2 text-violet">Record</label>
            </span>
            <span className="ms-8 text-lg">
              <input 
                checked={choice === "Upload"} 
                className="accent-violet-600 w-4 h-4" 
                type="radio" 
                id="upload" 
                name="type" 
                value="Upload" 
                onChange={(e) => setChoice(e.target.value)}
              />
              <label htmlFor="upload" className="ml-2 text-violet">Upload</label>
            </span> 
          </div>

          {/* File upload/record section outside of form */}
          <div className="flex w-full items-start justify-center bg-grey-lighter mb-5 mt-[5rem]">
            {choice === "Upload" ? (
              <label className="w-64 flex flex-col items-center px-4 py-6 bg-white text-blue rounded-lg shadow-lg tracking-wide uppercase border border-blue cursor-pointer hover:bg-blue hover:text-blue-600">
                <svg
                  className="w-8 h-8"
                  fill="blue"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                >
                  <path d="M16.88 9.1A4 4 0 0 1 16 17H5a5 5 0 0 1-1-9.9V7a3 3 0 0 1 4.52-2.59A4.98 4.98 0 0 1 17 8c0 .38-.04.74-.12 1.1zM11 11h3l-4-4-4 4h3v3h2v-3z" />
                </svg>
                <span className="mt-2 text-base leading-normal">Select a file</span>
                
                <input 
                  type="file"  
                  name="file"
                  accept="audio/*"
                  className="block w-full text-sm text-slate-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-full file:border-0
                    file:text-sm file:font-semibold
                    file:bg-violet-50 file:text-violet-700
                    hover:file:bg-violet-100" 
                  onChange={handleFileUpload}
                />
              </label>
            ) : (
              <div>
                <AudioRecorder onStop={handleRecordingStop} />
              </div>
            )}
          </div>
          
          <span className="text-white">File: {filename}</span>

          {/* Form only wraps the submit button */}
          <form onSubmit={handleSubmit}>
            <div className="flex items-center justify-center">
              <button 
                className="flex bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full mt-5" 
                type="submit"
              >
                PREDICT
              </button>
            </div>
          </form>

          <div className="mt-[5rem] mb-4 text-2xl">
            <span className="bg-gradient-to-r from-zinc-500 to-blue-500 text-transparent bg-clip-text">
              Detected Species is: {val}
            </span>
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r to-violet-600 from-blue-900 font-black">
              Confidence Score: {confidenceVal}
            </span>
          </div>
      </main>
      
      <div className="bg-gradient-to-r from-zinc-500 to-blue-500 w-full mb-0">
          <footer>
        <p>© 2018 Gandalf</p>
      </footer>
      </div>
      
    </div>
  );
}

export default App;