import { FC } from 'react';
import SensorData from '../components/SensorData';
import GameController from '../components/GameController';
import BandwidthTracker from '../components/BandwidthTracker';
import PowerTracker from '../components/PowerTracker';
import EnvStateTracker from '../components/EnvStateTracker';

const GameControllerPage: FC = () => {
    return (
        <div className="flex flex-col h-screen bg-red-50">
            <header className="bg-red-600 text-white p-4">
                <h1 className="text-2xl font-bold">Robot Control Dashboard</h1>
            </header>
            <main className="flex-grow flex p-4 space-x-4">
                <div className="w-2/3 bg-white rounded-lg shadow-lg p-4 border border-red-100">
                    <SensorData />
                    <h2 className="text-xl font-semibold mb-4 mt-8 text-gray-800">Bandwidth Usage</h2>
                    <BandwidthTracker />
                    <h2 className="text-xl font-semibold mb-4 mt-8 text-gray-800">Power Consumption</h2>
                    <PowerTracker />
                    <h2 className="text-xl font-semibold mb-4 mt-8 text-gray-800">Reinforcement Learning Visualization</h2>
                    <div className="w-full h-96 bg-red-50 border border-red-200 p-4 overflow-hidden">
                        <EnvStateTracker />
                    </div>
                </div>
                <div className="w-1/3 bg-white rounded-lg shadow-lg p-4 border border-red-100">
                    <h2 className="text-xl font-semibold mb-4 text-gray-800">Controls</h2>
                    <GameController />
                </div>
            </main>
        </div>
    );
};

export default GameControllerPage;
