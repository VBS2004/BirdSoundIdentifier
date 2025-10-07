import { useState, useEffect } from "react";
import { Bird, Search, Grid, List, Heart, Camera, ExternalLink, Download, Loader2, AlertCircle } from "lucide-react";

function SpeciesPage() {
  const [species, setSpecies] = useState([]);
  const [filteredSpecies, setFilteredSpecies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [viewMode, setViewMode] = useState("grid");
  const [selectedSpecies, setSelectedSpecies] = useState(null);
  const [filterCategory, setFilterCategory] = useState("all");
  const [showDownloadNotification, setShowDownloadNotification] = useState(false);
  const [selectedSpeciesImages, setSelectedSpeciesImages] = useState([]);
  const [imagesLoading, setImagesLoading] = useState(false);

  useEffect(() => {
    fetchSpecies();
  }, []);

  useEffect(() => {
    filterSpecies();
  }, [species, searchTerm, filterCategory]);

  useEffect(() => {
    if (selectedSpecies) {
      loadSpeciesImages(selectedSpecies.name);
    } else {
      setSelectedSpeciesImages([]);
    }
  }, [selectedSpecies]);

  const fetchSpecies = async () => {
    try {
      setLoading(true);
      const response = await fetch("http://localhost:5000/species");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log("Species data:", data);
      const speciesData = data.species || data || [];
      setSpecies(speciesData);
      setFilteredSpecies(speciesData);
    } catch (error) {
      console.error("Error fetching species:", error);
      setError("Failed to load species data. Please check your connection.");
    } finally {
      setLoading(false);
    }
  };

  const loadSpeciesImages = async (speciesName) => {
    setImagesLoading(true);
    try {
      const response = await fetch("http://localhost:5000/species/images", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ species_name: speciesName }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      console.log("Species images:", data);

      const imageList = data.image_urls.map((imageData, index) => ({
        id: index,
        url: imageData.medium,
        originalUrl: imageData.original,
        largeUrl: imageData.large,
        smallUrl: imageData.small,
        photographer: imageData.photographer,
        alt: imageData.alt || `${speciesName} - Image ${index + 1}`,
        pexelsId: imageData.id,
        loaded: false,
      }));

      setSelectedSpeciesImages(imageList);
    } catch (error) {
      console.error("Error loading species images:", error);
    } finally {
      setImagesLoading(false);
    }
  };

  const filterSpecies = () => {
    let filtered = species;
    if (searchTerm) {
      filtered = filtered.filter((s) =>
        s.name?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    setFilteredSpecies(filtered);
  };

  const handleDownload = async (image) => {
    try {
      const url = image.originalUrl || image.largeUrl || image.url || image.smallUrl;
      console.log("Download triggered for image:", image.id, "URL:", url);
      setShowDownloadNotification(true);
      const response = await fetch(url);
      if (!response.ok) throw new Error("Failed to fetch image");
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `${selectedSpecies?.name || 'species'}_${image.id}.${url.split('.').pop()}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
      setTimeout(() => setShowDownloadNotification(false), 2000);
    } catch (error) {
      console.error("Download error:", error);
      alert("Failed to download image: " + error.message);
      setShowDownloadNotification(false);
    }
  };

  const handleViewSource = (image) => {
    const sourceUrl = `https://www.pexels.com/photo/${image.pexelsId}/`;
    console.log("View Source triggered for image:", image.id, "URL:", sourceUrl);
    window.open(sourceUrl, "_blank");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white overflow-hidden relative">
      {showDownloadNotification && (
        <div className="fixed top-4 right-4 z-50 px-4 py-2 bg-green-500/90 text-white rounded-xl shadow-lg backdrop-blur-sm animate-slideInOut">
          Download Started
        </div>
      )}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 h-60 bg-indigo-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse delay-500"></div>
      </div>

      <main className="relative z-10 flex-1 max-w-7xl mx-auto px-6 py-12 pt-20">
        <div className="text-center mb-16">
          <h2 className="text-5xl md:text-7xl font-black mb-6 bg-gradient-to-r from-white via-blue-200 to-purple-200 bg-clip-text text-transparent leading-tight">
            Bird Species
            <br />
            <span className="text-4xl md:text-6xl">Database</span>
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed">
            Explore our comprehensive collection of bird species with detailed information and visual references
          </p>
        </div>

        <div className="mb-12">
          <div className="flex flex-col lg:flex-row gap-6 items-center justify-between">
            <div className="relative flex-1 max-w-2xl">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search species by name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-12 pr-4 py-4 bg-black/30 backdrop-blur-md border border-white/10 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-300"
              />
            </div>
            <div className="flex items-center space-x-4">
              <div className="bg-black/30 backdrop-blur-md rounded-xl p-1 border border-white/10">
                <div className="flex space-x-1">
                  {[{ mode: "grid", icon: Grid }, { mode: "list", icon: List }].map(
                    ({ mode, icon: Icon }) => (
                      <button
                        key={mode}
                        onClick={() => setViewMode(mode)}
                        className={`p-2 rounded-lg transition-all duration-300 ${
                          viewMode === mode
                            ? 'bg-blue-500 text-white'
                            : 'text-gray-400 hover:text-white hover:bg-white/10'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                      </button>
                    )
                  )}
                </div>
              </div>
            </div>
          </div>
          <div className="mt-6 text-center">
            <p className="text-gray-400">
              Showing {filteredSpecies.length} of {species.length} species
            </p>
          </div>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-400" />
              <p className="text-xl text-gray-300">Loading species database...</p>
              <p className="text-sm text-gray-500 mt-2">Fetching data from server</p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-center justify-center py-20">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 mx-auto mb-6 bg-red-500/20 rounded-2xl flex items-center justify-center">
                <AlertCircle className="w-8 h-8 text-red-400" />
              </div>
              <h3 className="text-2xl font-bold mb-4 text-red-400">Unable to Load Species</h3>
              <p className="text-gray-400 mb-6">{error}</p>
              <button
                onClick={fetchSpecies}
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl font-medium hover:from-blue-600 hover:to-purple-700 transition-all duration-300"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        {!loading && !error && (
          <>
            {filteredSpecies.length === 0 ? (
              <div className="text-center py-20">
                <div className="w-16 h-16 mx-auto mb-6 bg-gray-500/20 rounded-2xl flex items-center justify-center">
                  <Search className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-2xl font-bold mb-4">No Species Found</h3>
                <p className="text-gray-400">Try adjusting your search terms or filters</p>
              </div>
            ) : (
              <div
                className={
                  viewMode === "grid"
                    ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8"
                    : "space-y-6"
                }
              >
                {filteredSpecies.map((species, index) => (
                  <SpeciesCard
                    key={species.id || index}
                    species={species}
                    viewMode={viewMode}
                    onClick={() => setSelectedSpecies(species)}
                  />
                ))}
              </div>
            )}
          </>
        )}

        {selectedSpecies && (
          <SpeciesModal
            species={selectedSpecies}
            onClose={() => setSelectedSpecies(null)}
            selectedSpeciesImages={selectedSpeciesImages}
            handleDownload={handleDownload}
            handleViewSource={handleViewSource}
            imagesLoading={imagesLoading}
          />
        )}
      </main>

      <footer className="relative z-10 mt-20 border-t border-white/10 bg-black/20 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 py-8 text-center">
          <p className="text-gray-400">&copy; 2025 BirdID. Powered by advanced AI technology.</p>
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
        @keyframes slideInOut {
          0% { opacity: 0; transform: translateY(-10px); }
          10% { opacity: 1; transform: translateY(0); }
          90% { opacity: 1; transform: translateY(0); }
          100% { opacity: 0; transform: translateY(-10px); }
        }
        .animate-slideInOut {
          animation: slideInOut 2s ease-out forwards;
        }
      `}</style>
    </div>
  );
}

function SpeciesCard({ species, viewMode, onClick }) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const handleImageLoad = () => setImageLoaded(true);
  const handleImageError = () => setImageError(true);

  return (
    <div
      className="group relative bg-gradient-to-br from-black/40 to-black/20 backdrop-blur-md border border-white/10 rounded-2xl overflow-hidden cursor-pointer transform hover:scale-105 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/20"
      onClick={onClick}
    >
      <div className="aspect-square relative">
        {species.image && !imageError ? (
          <>
            <img
              src={species.image}
              alt={species.name}
              className={`w-full h-full object-cover transition-opacity duration-300 ${
                imageLoaded ? 'opacity-100' : 'opacity-0'
              }`}
              onLoad={handleImageLoad}
              onError={handleImageError}
            />
            {!imageLoaded && (
              <div className="absolute inset-0 bg-gradient-to-br from-gray-700 to-gray-600 flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
              </div>
            )}
          </>
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-gray-700 to-gray-600 flex items-center justify-center">
            <Bird className="w-16 h-16 text-gray-400" />
          </div>
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <div className="absolute bottom-4 left-4 right-4">
            <div className="flex items-center justify-between">
              <div className="flex space-x-2">
                <button className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/40 transition-colors">
                  <Heart className="w-4 h-4" />
                </button>
                <button className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/40 transition-colors">
                  <Camera className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="p-6">
        <h3 className="text-xl font-bold text-white mb-2 truncate">
          {species.name || 'Unknown Species'}
        </h3>
      </div>
    </div>
  );
}

function SpeciesModal({ species, onClose, selectedSpeciesImages, handleDownload, handleViewSource, imagesLoading }) {
  const [imageLoaded, setImageLoaded] = useState(false);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="relative max-w-4xl max-h-[90vh] bg-gradient-to-br from-black/60 to-black/40 backdrop-blur-md border border-white/20 rounded-3xl overflow-hidden">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 w-10 h-10 bg-black/50 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-black/70 transition-colors"
        >
          ✕
        </button>
        <div className="aspect-video relative">
          {imagesLoading ? (
            <div className="w-full h-full bg-gradient-to-br from-gray-700 to-gray-600 flex items-center justify-center">
              <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
            </div>
          ) : selectedSpeciesImages.length > 0 ? (
            <>
              <img
                src={selectedSpeciesImages[0].url}
                alt={selectedSpeciesImages[0].alt}
                className={`w-full h-full object-cover transition-opacity duration-300 ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
                onLoad={() => setImageLoaded(true)}
                onError={(e) => {
                  if (selectedSpeciesImages[0].smallUrl && e.target.src !== selectedSpeciesImages[0].smallUrl) {
                    e.target.src = selectedSpeciesImages[0].smallUrl;
                  } else {
                    e.target.style.display = 'none';
                  }
                }}
              />
              {!imageLoaded && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
                </div>
              )}
            </>
          ) : (
            <div className="w-full h-full bg-gradient-to-br from-gray-700 to-gray-600 flex items-center justify-center">
              <Bird className="w-16 h-16 text-gray-400" />
            </div>
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
        </div>
        <div className="p-8">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-4xl font-bold text-white mb-2">
                {species.name || 'Unknown Species'}
              </h2>
            </div>
          </div>
          <div className="flex space-x-4">
            <button
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl font-medium hover:from-blue-600 hover:to-purple-700 transition-all duration-300"
            >
              <Heart className="w-4 h-4" />
              <span>Add to Favorites</span>
            </button>
            {selectedSpeciesImages.length > 0 && (
              <>
                <button
                  onClick={() => handleDownload(selectedSpeciesImages[0])}
                  className="flex items-center space-x-2 px-6 py-3 bg-white/10 backdrop-blur-sm rounded-xl hover:bg-white/20 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  <span>Download Image</span>
                </button>
                <button
                  onClick={() => handleViewSource(selectedSpeciesImages[0])}
                  className="flex items-center space-x-2 px-6 py-3 bg-white/10 backdrop-blur-sm rounded-xl hover:bg-white/20 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span>View Source</span>
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SpeciesPage;