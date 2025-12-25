// components/panels/ExportPanel.jsx
export default function ExportPanel({ onExport }) {
  return (
    <div className="export-panel">
      <div className="nav-title">Экспорт</div>

      <div className="export-group">
        <div className="export-label">3D Модель</div>
        <div className="button-group">
          <button className="export-button" onClick={() => onExport("ply")}>PLY</button>
          <button className="export-button" onClick={() => onExport("stl")}>STL</button>
        </div>
      </div>

      <div className="export-group">
        <div className="export-label">Маска</div>
        <div className="button-group">
          <button className="export-button" onClick={() => onExport("nii")}>NIFTI (.nii)</button>
        </div>
      </div>

      <div className="export-group">
        <div className="export-label">Отчёт</div>
        <div className="button-group">
          <button className="export-button" onClick={() => onExport("json")}>JSON отчет</button>
        </div>
      </div>
    </div>
  );
}
