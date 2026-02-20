import axios from "axios";

const MIDDLEWARE_URL = "http://localhost:3000";

export async function predictImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(`${MIDDLEWARE_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
    });

    return response.data;
}
