import { useEffect, useState } from "react";
import { fetchSliceStack } from "/src/api/slices.js";
import SliceCanvas from "./SliceCanvas";

export default function SliceViewer({ maskPath, viewType }) {
  const [stack, setStack] = useState([]);
  const [index, setIndex] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!maskPath) return;

    setLoading(true);
    fetchSliceStack(maskPath, viewType)
      .then((data) => {
        setStack(data.slices);
        setIndex(Math.floor(data.slices.length / 2));
      })
      .finally(() => setLoading(false));
  }, [maskPath, viewType]);

  if (loading) return <div>Загрузка срезов…</div>;
  if (!stack.length) return null;

  return (
    <div className="w-full h-full flex flex-col">
      <SliceCanvas image={stack[index]} />

      <input
        type="range"
        min={0}
        max={stack.length - 1}
        value={index}
        onChange={(e) => setIndex(Number(e.target.value))}
        className="w-full mt-2"
      />

      <div className="text-sm text-center">
        Срез {index + 1} / {stack.length}
      </div>
    </div>
  );
}
