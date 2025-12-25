import exportIcon from '../../../assets/Download.svg';
import save_doc from '../../../assets/save_doc.svg';


export default function ExportPanel({ onExport }) {
  return (
    <div className="export-panel">
      <div className="export-title">
        <img 
          src={exportIcon} 
          alt=""
          className="export-main-icon"
        />
        Экспорт
      </div>

      <div className="export-group">
        <div className="export-label">
          <img 
            src={save_doc} 
            alt=""
            className="export-icons"
          />
          3D Модель
        </div>
        <div className="button-group">
          <button className="export-button-3d" onClick={() => onExport("ply")}>PLY</button>
          <button className="export-button-3d" onClick={() => onExport("stl")}>STL</button>
        </div>
      </div>

      <div className="export-group">
        <div className="export-label">
          <img 
            src={save_doc} 
            alt=""
            className="export-icons"
          />
          Маска
        </div>
        <div className="button-group">
          <button className="export-button-mask" onClick={() => onExport("nii")}>NIFTI (.nii)</button>
        </div>
      </div>

      <div className="export-group">
        <div className="export-label">
          <img 
            src={save_doc} 
            alt=""
            className="export-icons"
          />
          Отчёт
        </div>
        <div className="button-group">
          <button className="export-button-report" onClick={() => onExport("json")}>JSON отчет</button>
        </div>
      </div>
    </div>
  );
}
