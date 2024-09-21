import React from 'react';

interface Tutorial {
  title: string;
  description: string;
  videoUrl?: string;
  guideUrl?: string;
}

interface TutorialSectionProps {
  tutorials: Tutorial[];
}

const TutorialSection: React.FC<TutorialSectionProps> = ({ tutorials }) => {
  return (
    <div className="mb-8">
      <h2 className="text-2xl font-semibold mb-4">Tutorials</h2>
      <div className="space-y-6">
        {tutorials.map((tutorial, index) => (
          <div key={index} className="bg-gray-50 p-4 rounded-lg shadow">
            <h3 className="text-xl font-medium mb-2">{tutorial.title}</h3>
            <p className="text-gray-700 mb-2">{tutorial.description}</p>
            {tutorial.videoUrl && (
              <div className="aspect-w-16 aspect-h-9">
                <iframe
                  src={tutorial.videoUrl}
                  title={tutorial.title}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  className="w-full h-full rounded-md"
                ></iframe>
              </div>
            )}
            {tutorial.guideUrl && (
              <a
                href={tutorial.guideUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                View Guide
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TutorialSection;
