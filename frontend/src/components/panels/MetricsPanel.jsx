export default function MetricsPanel({ metrics }) {
  return (
    <div className="p-4 border-l border-zinc-800 space-y-4">
      <div>Объём: {metrics.volume_ml} мл</div>
      <div>Площадь: {metrics.surface_mm2} мм²</div>
      <div>Центр масс: {metrics.center_of_mass}</div>
    </div>
  );
}
