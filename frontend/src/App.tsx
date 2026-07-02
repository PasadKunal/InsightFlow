import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import DashboardPage from "./pages/DashboardPage";
import ExperimentPage from "./pages/ExperimentPage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/experiments/:id" element={<ExperimentPage />} />
      </Routes>
    </Layout>
  );
}
