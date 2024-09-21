import React from 'react';
import DPad from './DPad';
import Joystick from './Joystick';
import { useGamepad } from '../hooks/useGamepad';
import { useKeyboard } from '../hooks/useKeyboard';
import Button from './Button';

const GameController: React.FC = () => {
  const isActive = true; // This can be set based on your app's logic
  const activeActionsGamepad = useGamepad(isActive); // Get active actions from gamepad
  const activeActionsKeyboard = useKeyboard(isActive); // Get active actions from keyboard

  // Handle direction press from DPad
  const handleDirectionPress = (direction: 'up' | 'down' | 'left' | 'right') => {
    console.log(`Direction Pressed: ${direction}`);
  };

  // Handle direction release from DPad
  const handleDirectionRelease = (direction: 'up' | 'down' | 'left' | 'right') => {
    console.log(`Direction Released: ${direction}`);
  };

  // Handle joystick movement
  const handleJoystickMove = (direction: string) => {
    console.log(`Joystick Moved: ${direction}`);
  };

  return (
    <div className="game-controller">
      <div className="controller bg-gradient-to-br from-pink-500 to-orange-400 rounded-lg shadow-md p-4">
        <DPad 
          onDirectionPress={handleDirectionPress} 
          onDirectionRelease={handleDirectionRelease} 
          className="bg-gray-200 rounded-full p-4"
        />
        <Joystick onMove={handleJoystickMove}  />
        <div className="button-group flex justify-around mt-4">
          <Button className="action-button bg-yellow-400 hover:bg-yellow-500 text-white font-bold py-2 px-4 rounded">A</Button>
          <Button className="action-button bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded">B</Button>
          <Button className="action-button bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded">X</Button>
          <Button className="action-button bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded">Y</Button>
        </div>
        <div className="trigger-group flex justify-around mt-4">
          <Button className="trigger bg-yellow-400 hover:bg-yellow-500 text-white font-bold py-2 px-4 rounded">L</Button>
          <Button className="trigger bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded">R</Button>
        </div>
      </div>
      <div>
        <h3>Active Actions:</h3>
        <pre>{JSON.stringify([...activeActionsGamepad, ...activeActionsKeyboard], null, 2)}</pre>
      </div>
    </div>
  );
};

export default GameController;
