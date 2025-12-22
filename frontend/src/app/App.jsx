import { AppStateProvider, useAppState } from "./appState";
import UploadScreen from "../screens/UploadScreen";
import ViewerScreen from "../screens/ViewerScreen";

function AppContent() {
  const { phase } = useAppState();

  if (phase === "idle") return <UploadScreen />;
  if (phase === "processing") {
    return <div className="processing-screen">Обработка…</div>;
  }
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