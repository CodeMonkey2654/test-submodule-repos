// src/pages/HelpAndSupportPage.tsx

import React from 'react';
import FAQSection from '../components/FAQSection';
import DocumentationSection from '../components/DocumentationSection';
import TutorialSection from '../components/TutorialSection';
import ContactForm from '../components/ContactForm';
import { FAQ } from '../types';

const faqs: FAQ[] = [
  {
    question: 'How do I navigate the dashboard?',
    answer:
      'Use the navigation bar at the top to access different sections such as Analytics, Controls, and Help. Click on the respective links to navigate to each section.',
  },
  {
    question: 'How can I view telemetry data?',
    answer:
      'Navigate to the Analytics section to view real-time and historical telemetry data. Use the filters to adjust the time range and select specific metrics to display.',
  },
  {
    question: 'How do I control the rover?',
    answer:
      'Go to the Controls section where you can use the command interface to maneuver the rover. You can send commands based on your user role and expertise.',
  },
  // Add more FAQs as needed
];

const documentationLinks = [
  { title: 'User Guide', url: 'https://example.com/user-guide' },
  { title: 'API Documentation', url: 'https://example.com/api-docs' },
  { title: 'System Overview', url: 'https://example.com/system-overview' },
  // Add more documentation links as needed
];

const tutorials = [
  {
    title: 'Getting Started with the Dashboard',
    description:
      'Learn how to navigate the dashboard, access different sections, and customize your view to monitor the rover effectively.',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ', // Replace with actual tutorial video URLs
  },
  {
    title: 'Controlling the Rover',
    description:
      'Step-by-step guide on how to use the command interface to maneuver the rover manually or override AI decisions.',
    guideUrl: 'https://example.com/controlling-the-rover', // Replace with actual guide URLs
  },
  {
    title: 'Analyzing Telemetry Data',
    description:
      'Understand how to interpret telemetry data, use filters, and perform comparative analyses to assess rover performance.',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ', // Replace with actual tutorial video URLs
  },
  // Add more tutorials as needed
];

const HelpPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <header className="mb-6">
        <h1 className="text-4xl font-bold text-center">Help & Support</h1>
      </header>
      <main className="max-w-5xl mx-auto space-y-8">
        {/* Documentation Section */}
        <DocumentationSection documentationLinks={documentationLinks} />

        {/* Tutorials Section */}
        <TutorialSection tutorials={tutorials} />

        {/* FAQ Section */}
        <FAQSection faqItems={faqs} />

        {/* Contact Form Section */}
        <ContactForm />
      </main>
    </div>
  );
};

export default HelpPage;
