import React from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

import App from './App.jsx'
import './index.css'

import "@mantine/core/styles.css";
import { MantineProvider } from "@mantine/core";

import 'katex/dist/katex.min.css';

import BlogRoot from './pages/BlogRoot'
import MixtureOfExperts from './pages/MixtureOfExperts';

const router = createBrowserRouter([
  {
    path: "/",
    element: <BlogRoot />
  },
  {
    path: "/mixture-of-experts-intro",
    element: <MixtureOfExperts />,
  },
]);

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <MantineProvider defaultColorScheme="dark">
      <RouterProvider router={router} />
    </MantineProvider>
  </React.StrictMode>,
)
