import { useLoader } from "@react-three/fiber";
import { STLLoader } from "three-stdlib";
import { Suspense } from "react";

function Model({ url }) {
  console.log("Loading model from:", url); // Отладка
  const geometry = useLoader(STLLoader, url);
  
  // STL файлы часто огромные - нужен большой scale
  return (
    <mesh geometry={geometry} scale={0.01} position={[0, 0, 0]}>
      <meshStandardMaterial color="#34d399" wireframe={true} /> // Сначала wireframe
    </mesh>
  );
}

export default function LiverMesh({ url }) {
  if (!url) return null;
  return (
    <Suspense fallback={<Fallback />}>
      <Model url={url} />
    </Suspense>
  );
}

function Fallback() {
  return (
    <mesh>
      <boxGeometry args={[10, 10, 10]} />
      <meshStandardMaterial color="gray" />
    </mesh>
  );
}