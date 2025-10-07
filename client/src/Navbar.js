import { Bird } from "lucide-react";

function Navbar({ isServerOnline }) {
  return (
    <header className="fixed top-0 left-0 w-full z-50 backdrop-blur-md bg-black/20 border-b border-white/10">
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
            ['Dashboard', '/'],
            ['Species', '/species'],
            ['Analytics', '/analytics'],
            ['Community', '/community'],
          ].map(([title, url]) => (
            <a
              key={title}
              href={url}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 backdrop-blur-sm ${
                window.location.pathname === url
                  ? 'text-white bg-white/20 border border-white/20'
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              {title}
            </a>
          ))}
        </nav>

        <div className="flex items-center space-x-2">
          <div
            className={`w-2 h-2 rounded-full animate-pulse ${
              isServerOnline ? 'bg-green-400' : 'bg-red-400'
            }`}
          ></div>
          <span
            className={`text-xs hidden sm:block ${
              isServerOnline ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {isServerOnline ? 'AI Online' : 'AI Offline'}
          </span>
        </div>
      </div>
    </header>
  );
}

export default Navbar;