import { createContext, useContext, useState } from "react";

const AppStateContext = createContext(null);

export function AppStateProvider({ children }) {
  const [phase, setPhase] = useState("idle"); 
  const [view, setView] = useState("3d");     // 3d | axial | sagittal | coronal
  const [meshData, setMeshData] = useState(null);

  return (
    <AppStateContext.Provider
      value={{ phase, setPhase, view, setView, meshData, setMeshData }}
    >
      {children}
    </AppStateContext.Provider>
  );
}

export function useAppState() {
  return useContext(AppStateContext);
}