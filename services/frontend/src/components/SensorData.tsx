import { FC, useEffect, useRef, useState } from 'react';

const SensorData: FC = () => {
    const [depthData, setDepthData] = useState<number[][]>([]);
    const [lidarData, setLidarData] = useState<number[]>([]);
    const [cubeRotation, setCubeRotation] = useState({ x: 0, y: 0, z: 0 });
  
    useEffect(() => {
      // Simulating sensor data updates
      const interval = setInterval(() => {
        // Generate rotating cube data
        const newDepthData = generateRotating3DCubeData(cubeRotation);
        setDepthData(newDepthData);
        setCubeRotation((prevRotation) => ({
          x: (prevRotation.x + 0.05) % (2 * Math.PI),
          y: (prevRotation.y + 0.07) % (2 * Math.PI),
          z: (prevRotation.z + 0.03) % (2 * Math.PI)
        }));
  
        // Generate LIDAR data from the same rotating cube data
        const newLidarData = generateLidarDataFromDepth(newDepthData);
        setLidarData(newLidarData);
      }, 1000);
  
      return () => clearInterval(interval);
    }, [cubeRotation]);
  
    const depthCanvas = useRef<HTMLCanvasElement>(null);
    const lidarCanvas = useRef<HTMLCanvasElement>(null);
  
    useEffect(() => {
      if (depthCanvas.current && depthData.length > 0) {
        const ctx = depthCanvas.current.getContext('2d');
        if (ctx) {
          const canvasWidth = depthCanvas.current.width;
          const canvasHeight = depthCanvas.current.height;
          const cellWidth = canvasWidth / 32;
          const cellHeight = canvasHeight / 32;
          depthData.forEach((row, y) => {
            row.forEach((depth, x) => {
              const color = Math.floor(255 - (depth / 10) * 255);
              ctx.fillStyle = `rgb(${color},${color},${color})`;
              ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
            });
          });
        }
      }
    }, [depthData]);
  
    useEffect(() => {
      if (lidarCanvas.current && lidarData.length > 0) {
        const ctx = lidarCanvas.current.getContext('2d');
        if (ctx) {
          const canvasWidth = lidarCanvas.current.width;
          const canvasHeight = lidarCanvas.current.height;
          const centerX = canvasWidth / 2;
          const centerY = canvasHeight / 2;
          const maxRadius = Math.min(centerX, centerY) * 0.9;
  
          ctx.fillStyle = 'black';
          ctx.fillRect(0, 0, canvasWidth, canvasHeight);
          ctx.beginPath();
          lidarData.forEach((distance, angle) => {
            const x = centerX + Math.cos(angle * Math.PI / 180) * (distance / 10 * maxRadius);
            const y = centerY + Math.sin(angle * Math.PI / 180) * (distance / 10 * maxRadius);
            ctx.lineTo(x, y);
          });
          ctx.closePath();
          ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
          ctx.fill();
        }
      }
    }, [lidarData]);
  
    const generateRotating3DCubeData = (rotation: { x: number, y: number, z: number }): number[][] => {
      const size = 32;
      const cubeSize = 16;
      const centerX = size / 2;
      const centerY = size / 2;
      const centerZ = size / 2;
      const data: number[][] = Array(size).fill(0).map(() => Array(size).fill(10));
    
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const dx = x - centerX;
          const dy = y - centerY;
          
          // Apply 3D rotation
          const rotatedX = dx * Math.cos(rotation.y) * Math.cos(rotation.z) - dy * Math.sin(rotation.z);
          const rotatedY = dx * Math.sin(rotation.x) * Math.sin(rotation.y) + dy * Math.cos(rotation.x) * Math.cos(rotation.z) + centerZ * Math.sin(rotation.x) * Math.cos(rotation.y);
          const rotatedZ = dx * Math.cos(rotation.x) * Math.sin(rotation.y) - dy * Math.sin(rotation.x) * Math.cos(rotation.z) + centerZ * Math.cos(rotation.x) * Math.cos(rotation.y);
    
          if (
            Math.abs(rotatedX) < cubeSize / 2 &&
            Math.abs(rotatedY) < cubeSize / 2 &&
            Math.abs(rotatedZ) < cubeSize / 2
          ) {
            const depth = 5 + 5 * Math.sin(rotation.x) * Math.cos(rotation.y) * Math.cos(rotation.z);
            data[y][x] = depth;
          }
        }
      }
    
      return data;
    };
  
    const generateLidarDataFromDepth = (depthData: number[][]): number[] => {
      const lidarData: number[] = [];
      const size = depthData.length;
      const center = size / 2;
    
      for (let angle = 0; angle < 360; angle++) {
        const radians = angle * Math.PI / 180;
        let x = center;
        let y = center;
        let distance = 0;
    
        while (x >= 0 && x < size && y >= 0 && y < size) {
          const depth = depthData[Math.floor(y)][Math.floor(x)];
          if (depth < 10) {
            distance = Math.sqrt((x - center) ** 2 + (y - center) ** 2);
            break;
          }
          x += Math.cos(radians);
          y += Math.sin(radians);
        }
    
        lidarData.push(distance || 10);
      }
    
      return lidarData;
    };
  
    return (
      <div className="mb-6 p-4 bg-gray-100 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">Sensor Data</h3>
        <div className="flex flex-col md:flex-row justify-between">
          <div className="w-full md:w-[48%] mb-4 md:mb-0">
            <h4 className="font-medium">3D Camera Depth (Rotating 3D Cube)</h4>
            <canvas ref={depthCanvas} width="100%" height="100%" className="w-full aspect-square border border-gray-300" />
          </div>
          <div className="w-full md:w-[48%]">
            <h4 className="font-medium">Lidar View (Rotating 3D Cube)</h4>
            <canvas ref={lidarCanvas} width="100%" height="100%" className="w-full aspect-square border border-gray-300" />
          </div>
        </div>
      </div>
    );
  };

  export default SensorData;