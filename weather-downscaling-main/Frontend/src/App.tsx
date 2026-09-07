'use client';

import { Navbar } from './components/nav/Navbar';
import { HomePage } from './views/HomePage';

export default function App() {
  return (
    <>
      <Navbar />
      <main>
        <HomePage />
      </main>
    </>
  );
}
