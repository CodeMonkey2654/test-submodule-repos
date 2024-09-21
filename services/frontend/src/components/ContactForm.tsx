import React, { useState } from 'react';
import { Card, Input, Textarea, Button, Alert, Spinner } from './index';
import { ContactFormData } from '../types';

const ContactForm: React.FC = () => {
  const [formData, setFormData] = useState<ContactFormData>({
    name: '',
    email: '',
    subject: '',
    message: '',
  });

  const [submitted, setSubmitted] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      // Replace with actual API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      console.log('Form submitted:', formData);
      setSubmitted(true);
      setFormData({ name: '', email: '', subject: '', message: '' });
    } catch (err) {
      setError('Failed to submit the form. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <Card title="Contact Support">
        <Alert type="success" message="Your message has been sent successfully!" />
      </Card>
    );
  }

  return (
    <Card title="Contact Support">
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && <Alert type="error" message={error} />}
        <Input
          label="Name"
          id="name"
          name="name"
          required
          value={formData.name}
          onChange={handleChange}
        />
        <Input
          label="Email"
          id="email"
          name="email"
          type="email"
          required
          value={formData.email}
          onChange={handleChange}
        />
        <Input
          label="Subject"
          id="subject"
          name="subject"
          required
          value={formData.subject}
          onChange={handleChange}
        />
        <Textarea
          label="Message"
          id="message"
          name="message"
          required
          rows={4}
          value={formData.message}
          onChange={handleChange}
        />
        <div className="flex items-center">
          <Button type="submit" variant="primary" disabled={loading}>
            {loading ? <Spinner /> : 'Submit'}
          </Button>
        </div>
      </form>
    </Card>
  );
};

export default ContactForm;
