# Bird Species Database

Welcome to the Bird Species Database, a web application that allows you to explore a comprehensive collection of bird species with detailed information and images sourced from the Pexels API. This project features a React frontend and a Flask backend with machine learning predictions and Redis caching for efficient image handling.

## Project Structure

- `client/`: Contains the React frontend.
  - Run with `npm start` to start the development server.
- `Server.py`: The Python Flask backend, located outside the `client` directory.
  - Runs the ML server and handles API requests, including Pexels image fetching and Redis caching.

## Prerequisites

### Frontend (client/)

- **Node.js**: Version 14.x or higher.
  - Install from nodejs.org.
- **npm**: Comes with Node.js.

### Backend

- **Python**: Version 3.8 or higher.
- **Redis**: For caching Pexels API responses.
  - Install Redis server (see below).
- **Required Python Packages**:
  - `flask`
  - `flask-cors`
  - `requests`
  - `redis`
  - `pillow` (for image validation)
  - Custom `predict` module (ensure `predict.py` is available).

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Set Up the Backend

- Navigate to the directory containing `Server.py` (outside `client/`).

- Install Python dependencies:

  ```bash
  pip install flask flask-cors requests redis pillow
  ```

- **Install and Start Redis**:

  - **Linux (Ubuntu)**:

    ```bash
    sudo apt update
    sudo apt install redis-server
    sudo systemctl enable redis
    sudo systemctl start redis
    ```

  - **MacOS**:

    ```bash
    brew install redis
    brew services start redis
    ```

  - **Windows**: Use WSL2 or Docker:

    ```bash
    docker run -d --name redis -p 6379:6379 redis:latest
    ```

  - Verify Redis:

    ```bash
    redis-cli ping
    ```

    Should return `PONG`.

- Configure your Pexels API key in `Server.py` (replace the placeholder `PEXELS_API_KEY`).

### 3. Set Up the Frontend

- Navigate to the `client/` directory:

  ```bash
  cd client
  ```

- Install Node dependencies:

  ```bash
  npm install
  ```

## Running the Application

### 1. Start the Backend

- From the directory containing `Server.py`:

  ```bash
  python Server.py
  ```

- The server will run on `http://localhost:5000`.

### 2. Start the Frontend

- From the `client/` directory:

  ```bash
  npm start
  ```

- The app will open in your browser at `http://localhost:3000`.

### 3. Verify

- Ensure the backend is running before starting the frontend.
- Visit `http://localhost:3000` to see the Bird Species Database.
- Switch between grid and list views, click a species to view images, and test download/view source functionality.

## Usage

- **Home Page**: Displays a list of bird species in grid or list view.
  - Grid View: Shows a `<Bird>` icon placeholder per species.
  - List View: Shows only species names.
- **Search**: Use the search bar to filter species by name.
- **Species Modal**: Click a species to open a modal with Pexels images, downloadable with attribution.
- **Features**:
  - Images fetched on demand via `/species/images` with Redis caching (24-hour TTL).
  - ML predictions for uploaded images via `/upload`.

## Screenshots

Here are some visuals of the application:

- **Home Page (Grid View)**:

  ![Grid View](screenshots/grid-view.png)*Shows the grid layout with bird placeholders.*

- **Home Page (List View)**:

  ![List View](screenshots/list-view.png)*Displays the text-only list view of species names.*

- **Species Modal**:

  ![Species Modal](screenshots/species-modal.png)*Shows the modal with an image, download, and view source options.*

*Note*: Replace `screenshots/*.png` with actual image files. To add screenshots:

1. Take screenshots of the app (e.g., using Snipping Tool or browser dev tools).
2. Save them in a `screenshots/` folder in the project root.
3. Update the `![...](...)` paths with the correct filenames (e.g., `screenshots/grid-view.jpg`).

## Contributing

Feel free to submit issues or pull requests. Please ensure:

- Code follows the project’s structure.
- Changes are tested with both frontend and backend running.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Pexels API for providing free bird images.
- Redis for caching support.
