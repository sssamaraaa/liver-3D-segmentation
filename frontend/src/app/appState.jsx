import { createContext, useContext, useState } from "react";

const AppStateContext = createContext(null);

export function AppStateProvider({ children }) {
  const [phase, setPhase] = useState("idle");     // idle | processing | done
  const [view, setView] = useState("3d");         // 3d | axial | sagittal | coronal
  const [meshData, setMeshData] = useState(null);
  const [progress, setProgress] = useState(0);   

  return (
    <AppStateContext.Provider
      value={{ phase, setPhase, view, setView, meshData, setMeshData, progress, setProgress }}
    >
      {children}
    </AppStateContext.Provider>
  );
}

export function useAppState() {
  return useContext(AppStateContext);
}
