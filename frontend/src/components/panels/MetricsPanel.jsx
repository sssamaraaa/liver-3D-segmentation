import metric from '../../../assets/Metriki.svg';
import circle from '../../../assets/Circle.svg';
import cube from '../../../assets/Cube.svg'; 
import size from '../../../assets/Size.svg';


export default function MetricsPanel({ metrics }) {
  return (
    <div className="metrics">

      <div className='metrics-title'>
        <img 
          src={metric} 
          alt=""
          className="metrics-main-icon"
        />
        Метрики
      </div>

      <div className="metric-titles">
        <img 
          src={cube} 
          alt=""
          className="metric-icons"
        />
        Объём: 
          <div className='metric-values'>
            {metrics.volume_ml} мл
          </div>
      </div>

      <div className='metric-titles'>
        <img 
          src={size} 
          alt=""
          className="metric-icons"
        />
        Площадь поверхности: 
          <div className='metric-values'>
            {metrics.surface_mm2} мм²
          </div>
      </div>

      <div className='metric-titles'>
        <div className="center-of-mass-container">
          <div className="axis-labels">
            <img src={circle} alt="" className="metric-icons" />
            Центр масс:
            <div className="axes-row">
              <span className="axis-label">X</span>
              <span className="axis-label">Y</span>
              <span className="axis-label">Z</span>
            </div>
          </div>
          <div className="values-row">
            {metrics.center_of_mass?.map((value, index) => (
              <span key={index} className="axis-value">
                {value?.toFixed(2) || '0.00'}
              </span>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}
