import { useState } from "react";
import UploadScreen from "./components/UploadScreen";
import ConfigureScreen from "./components/ConfigureScreen";
import ResultsScreen from "./components/ResultsScreen";
import { UploadResponse, PostmortemResponse } from "./types";
import "./App.css";

type Screen = "upload" | "configure" | "results";

function App() {
  const [screen, setScreen] = useState<Screen>("upload");
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [results, setResults] = useState<PostmortemResponse | null>(null);

  const handleUploadSuccess = (data: UploadResponse) => {
    setUploadData(data);
    setScreen("configure");
  };

  const handleAnalysisSuccess = (data: PostmortemResponse) => {
    setResults(data);
    setScreen("results");
  };

  const handleStartOver = () => {
    setUploadData(null);
    setResults(null);
    setScreen("upload");
  };

  return (
    <div className="app">
      {screen === "upload" && (
        <UploadScreen onSuccess={handleUploadSuccess} />
      )}
      {screen === "configure" && uploadData && (
        <ConfigureScreen
          uploadData={uploadData}
          onSuccess={handleAnalysisSuccess}
        />
      )}
      {screen === "results" && results && (
        <ResultsScreen results={results} onStartOver={handleStartOver} />
      )}
    </div>
  );
}

export default App;