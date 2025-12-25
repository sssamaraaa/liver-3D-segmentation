import { AppStateProvider, useAppState } from "./appState";
import UploadScreen from "../screens/UploadScreen";
import ViewerScreen from "../screens/ViewerScreen";
import ProcessingScreen from "../screens/ProcessingScreen";

function AppContent() {
  const { phase, progress } = useAppState();

  if (phase === "idle") return <UploadScreen />;
  if (phase === "processing") return <ProcessingScreen progress={progress} />;

  return <ViewerScreen />;
}

export default function App() {
  return (
    <AppStateProvider>
      <div className="app-container">
        <AppContent />
      </div>
    </AppStateProvider>
  );
}
